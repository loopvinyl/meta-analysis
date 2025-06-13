import pandas as pd
import streamlit as st
import altair as alt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np

# --- 1. Carregamento de Dados ---
# O arquivo dados_vermicomposto.xlsx deve estar na mesma pasta que este script.
try:
    df = pd.read_excel('dados_vermicomposto.xlsx')
except FileNotFoundError:
    st.error("Erro: Arquivo 'dados_vermicomposto.xlsx' não encontrado. Por favor, certifique-se de que ele está na mesma pasta que este script.")
    st.stop()

# Converte 'Material_Group' para categoria para garantir tratamento correto
# Assumindo que a coluna de agrupamento se chama 'Material_Group'. Ajuste se for diferente.
if 'Material_Group' in df.columns:
    df['Material_Group'] = df['Material_Group'].astype('category')
else:
    st.error("Erro: Coluna 'Material_Group' não encontrada no arquivo de dados. Por favor, renomeie sua coluna de agrupamento ou ajuste o código.")
    st.stop()

# --- 2. Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Metanálise de Vermicompostos",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Análise Metanalítica de Vermicompostos por Tipo de Resíduo")
st.markdown("Este aplicativo interativo permite explorar os resultados dos testes estatísticos de Nitrogênio, Fósforo, Potássio, pH e Razão C/N em vermicompostos, agrupados por material de origem.")
st.markdown("---")

# --- 3. Sidebar para Seleção de Variáveis ---
st.sidebar.header("Configurações da Análise")
variavel_selecionada = st.sidebar.selectbox(
    "Selecione a Variável para Análise:",
    ['N_perc', 'P_perc', 'K_perc', 'pH_final', 'C_N_Ratio_final'],
    format_func=lambda x: {
        'N_perc': 'Nitrogênio (%)',
        'P_perc': 'Fósforo (%)',
        'K_perc': 'Potássio (%)',
        'pH_final': 'pH',
        'C_N_Ratio_final': 'Razão C/N'
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.info("Os testes são realizados automaticamente. Os resultados exibem os p-valores para a homogeneidade de variâncias (Levene), normalidade (Shapiro-Wilk) e a comparação entre grupos (ANOVA ou Kruskal-Wallis).")

# --- 4. Função para realizar e exibir os testes estatísticos ---
def run_statistical_analysis(data, var_name, group_col):
    st.subheader(f"Resultados para: **{variavel_selecionada.replace('_', ' ').replace('perc', '%').replace('final', '')}**")

    # Remove NaNs para os testes
    data_clean = data[[var_name, group_col]].dropna()

    if data_clean.empty:
        st.warning(f"Não há dados suficientes para a variável '{var_name}' após remover valores ausentes.")
        return

    groups = data_clean[group_col].unique()
    num_groups = len(groups)

    if num_groups < 2:
        st.warning(f"Apenas {num_groups} grupo(s) de material. Não é possível realizar testes de comparação entre grupos.")
        st.markdown("#### Visualização dos Dados (Boxplot)")
        chart = alt.Chart(data_clean).mark_boxplot(size=50).encode(
            x=alt.X(group_col + ':N', title="Grupo de Material"),
            y=alt.Y(var_name + ':Q', title=f"{variavel_selecionada.replace('_', ' ').replace('perc', '%').replace('final', '')}"),
            tooltip=[group_col, alt.Tooltip(var_name, title=variavel_selecionada.replace('_', ' ').replace('perc', '%').replace('final', ''))]
        ).properties(
            title=f"Distribuição de {variavel_selecionada.replace('_', ' ').replace('perc', '%').replace('final', '')} por Grupo de Material"
        )
        st.altair_chart(chart, use_container_width=True)
        return

    st.markdown(f"**Número total de observações:** {len(data_clean)}")
    st.markdown(f"**Número de grupos de materiais:** {num_groups}")

    # --- Teste de Levene (Homogeneidade de Variâncias) ---
    st.markdown("#### Teste de Levene (Homogeneidade de Variâncias)")
    try:
        # Filtra grupos com dados suficientes para o teste de Levene (mínimo 2 observações por grupo)
        levene_data = []
        valid_groups_for_levene = []
        for g in groups:
            group_data = data_clean[var_name][data_clean[group_col] == g].dropna()
            if len(group_data) >= 2:
                levene_data.append(group_data)
                valid_groups_for_levene.append(g)

        if len(levene_data) >= 2: # Levene precisa de pelo menos 2 grupos com dados
            stat_levene, p_levene = stats.levene(*levene_data)
            st.write(f"Estatística de Levene: {stat_levene:.4f}, p-valor: {p_levene:.4f}")
            if p_levene >= 0.05:
                st.success("✅ As variâncias são homogêneas (p-valor ≥ 0.05).")
                homogeneity_met = True
            else:
                st.error("❌ As variâncias NÃO são homogêneas (p-valor < 0.05).")
                homogeneity_met = False
        else:
            st.warning("Não há grupos suficientes com 2 ou mais observações para o Teste de Levene.")
            homogeneity_met = False
    except ValueError as e:
        st.warning(f"Não foi possível realizar o Teste de Levene: {e}.")
        homogeneity_met = False

    # --- Teste de Shapiro-Wilk (Normalidade por Grupo) ---
    st.markdown("#### Teste de Shapiro-Wilk (Normalidade por Grupo)")
    normal_met_all_groups = True
    for g in groups:
        group_data = data_clean[var_name][data_clean[group_col] == g].dropna()
        if len(group_data) >= 3: # Shapiro-Wilk requer no mínimo 3 pontos
            stat_shapiro, p_shapiro = stats.shapiro(group_data)
            st.write(f"  - Grupo '{g}': Estatística de Shapiro: {stat_shapiro:.4f}, p-valor: {p_shapiro:.4f}")
            if p_shapiro < 0.05:
                normal_met_all_groups = False
        else:
            st.info(f"  - Grupo '{g}': Poucas observações ({len(group_data)}). Teste de Shapiro-Wilk não aplicável ou confiável.")

    if normal_met_all_groups:
        st.success("✅ A maioria dos grupos (com N>=3) apresenta normalidade. Pressuposto de normalidade satisfeito.")
    else:
        st.error("❌ Pelo menos um grupo (com N>=3) não atende ao pressuposto de normalidade.")


    # --- Decisão sobre o Teste Principal ---
    if homogeneity_met and normal_met_all_groups:
        st.markdown("#### Teste Principal: ANOVA Paramétrica")
        test_type = "parametric"
    else:
        st.markdown("#### Teste Principal: Kruskal-Wallis (Não Paramétrico)")
        test_type = "non_parametric"

    # --- Execução do Teste Principal ---
    if test_type == "parametric":
        try:
            model = ols(f'{var_name} ~ C({group_col})', data=data_clean).fit()
            from statsmodels.stats.anova import anova_lm
            anova_result = anova_lm(model, typ=2)
            st.write(anova_result)
            p_anova = anova_result['PR(>F)'][group_col]

            if p_anova < 0.05:
                st.success(f"**Resultado ANOVA:** p-valor = {p_anova:.4f} (significativo). Existem diferenças significativas entre os grupos.")
                st.markdown("##### Teste Post-hoc (Tukey HSD)")
                try:
                    tukey_result = pairwise_tukeyhsd(endog=data_clean[var_name], groups=data_clean[group_col], alpha=0.05)
                    st.write(tukey_result.summary())
                except ValueError as e:
                    st.warning(f"Não foi possível executar Tukey HSD: {e}. Pode ser devido a grupos com apenas uma observação após a limpeza de NaNs.")
            else:
                st.info(f"**Resultado ANOVA:** p-valor = {p_anova:.4f} (NÃO significativo). Não há diferenças significativas entre os grupos.")
        except Exception as e:
            st.error(f"Erro ao executar ANOVA: {e}. Verifique a estrutura dos dados e a quantidade de observações por grupo.")
    else: # non_parametric
        try:
            group_data_for_kw = [data_clean[var_name][data_clean[group_col] == g].values for g in groups if len(data_clean[var_name][data_clean[group_col] == g].values) > 0]
            
            if len(group_data_for_kw) < 2:
                st.warning("Não há grupos suficientes com dados para o Teste de Kruskal-Wallis.")
            else:
                stat_kw, p_kw = stats.kruskal(*group_data_for_kw)
                st.write(f"Estatística de Kruskal-Wallis: {stat_kw:.4f}, p-valor: {p_kw:.4f}")

                if p_kw < 0.05:
                    st.success(f"**Resultado Kruskal-Wallis:** p-valor = {p_kw:.4f} (significativo). Existem diferenças significativas entre os grupos.")
                    st.markdown("##### Teste Post-hoc (Teste de Dunn com correção de Bonferroni)")
                    try:
                        dunn_result = sp.posthoc_dunn(data_clean, val_col=var_name, group_col=group_col, p_adjust='bonferroni')
                        st.write(dunn_result)
                    except Exception as e:
                        st.warning(f"Não foi possível executar o Teste de Dunn: {e}. Verifique se há dados suficientes em cada grupo para comparações.")
                else:
                    st.info(f"**Resultado Kruskal-Wallis:** p-valor = {p_kw:.4f} (NÃO significativo). Não há diferenças significativas entre os grupos.")
        except Exception as e:
            st.error(f"Erro ao executar Kruskal-Wallis/Dunn: {e}. Verifique a estrutura dos dados ou a quantidade de observações por grupo.")

    # --- Visualização (Boxplot) ---
    st.markdown("#### Visualização dos Dados (Boxplot)")

    chart = alt.Chart(data_clean).mark_boxplot(size=50).encode(
        x=alt.X(group_col + ':N', title="Grupo de Material"),
        y=alt.Y(var_name + ':Q', title=f"{variavel_selecionada.replace('_', ' ').replace('perc', '%').replace('final', '')}"),
        tooltip=[group_col, alt.Tooltip(var_name, title=variavel_selecionada.replace('_', ' ').replace('perc', '%').replace('final', ''))]
    ).properties(
        title=f"Distribuição de {variavel_selecionada.replace('_', ' ').replace('perc', '%').replace('final', '')} por Grupo de Material"
    )
    st.altair_chart(chart, use_container_width=True)

# --- 5. Executar a Análise ---
if variavel_selecionada:
    run_statistical_analysis(df, variavel_selecionada, 'Material_Group')

# --- 6. Rodapé ---
st.markdown("---")
st.caption("Desenvolvido para a metanálise de vermicompostos. Por favor, consulte o artigo científico para detalhes metodológicos completos.")
