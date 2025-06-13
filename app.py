#chatjpt
import pandas as pd
import streamlit as st
import altair as alt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np

# --- Função de Compact Letter Display (CLD) ---
def get_compact_letter_display(p_values_matrix, group_names):
    sorted_groups = sorted(group_names)
    significant_pairs = []
    if not isinstance(p_values_matrix, pd.DataFrame):
        st.error("Internal Error: P-value matrix for CLD is not in the expected format (DataFrame).")
        return {group: '' for group in group_names}
    
    for i in range(len(sorted_groups)):
        for j in range(i + 1, len(sorted_groups)):
            g1 = sorted_groups[i]
            g2 = sorted_groups[j]
            if g1 in p_values_matrix.index and g2 in p_values_matrix.columns:
                p_val = p_values_matrix.loc[g1, g2]
            elif g2 in p_values_matrix.index and g1 in p_values_matrix.columns:
                p_val = p_values_matrix.loc[g2, g1]
            else:
                p_val = 1.0
            if p_val < 0.05:
                significant_pairs.append(tuple(sorted(tuple([g1, g2]))))
    group_letters = {g: [] for g in group_names}
    clusters = []
    for g in sorted_groups:
        assigned = False
        for cluster in clusters:
            can_add = True
            for member in cluster:
                if tuple(sorted((g, member))) in significant_pairs:
                    can_add = False
                    break
            if can_add:
                cluster.add(g)
                assigned = True
                break
        if not assigned:
            clusters.append({g})
    for i, cluster in enumerate(clusters):
        letter = chr(ord('a') + i)
        for group in cluster:
            group_letters[group].append(letter)
    final_letters = {group: "".join(sorted(list(set(letters)))) for group, letters in group_letters.items()}
    return final_letters

# --- Função de análise estatística e gráfico ---
def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    st.markdown(f"#### Análise para: **{dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}**")

    data_clean = data.dropna(subset=[dependent_var_name, group_var_name]).copy()

    if data_clean[group_var_name].nunique() < 2:
        st.warning(f"Não há grupos suficientes ({data_clean[group_var_name].nunique()}) para análise estatística de {dependent_var_name}.")
        return

    # Teste de homogeneidade (Levene)
    with st.expander(f"Teste de Homogeneidade de Variância (Levene) para {dependent_var_name}"):
        try:
            groups = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() for g in data_clean[group_var_name].unique()]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                st.info("Grupos insuficientes para o teste de Levene.")
                homogeneous_variances = False
            else:
                stat, p_levene = stats.levene(*groups)
                st.write(f"Levene Estatística: {stat:.3f}, p-valor: {p_levene:.3f}")
                homogeneous_variances = p_levene >= 0.05
                if homogeneous_variances:
                    st.success("Variâncias homogêneas (p >= 0.05).")
                else:
                    st.warning("Variâncias NÃO homogêneas (p < 0.05).")
        except Exception as e:
            st.error(f"Erro no teste de Levene: {e}")
            homogeneous_variances = False

    # Teste de normalidade (Shapiro-Wilk)
    with st.expander(f"Teste de Normalidade (Shapiro-Wilk) por grupo para {dependent_var_name}"):
        normality_by_group = True
        shapiro_results = []
        for group in data_clean[group_var_name].unique():
            group_data = data_clean[data_clean[group_var_name] == group][dependent_var_name].dropna()
            if len(group_data) >= 3:
                stat_shapiro, p_shapiro = stats.shapiro(group_data)
                shapiro_results.append({'Grupo': group, 'N': len(group_data), 'Estatística': stat_shapiro, 'p-valor': p_shapiro})
                if p_shapiro < 0.05:
                    normality_by_group = False
            else:
                shapiro_results.append({'Grupo': group, 'N': len(group_data), 'Estatística': np.nan, 'p-valor': np.nan})
                st.info(f"Grupo '{group}' tem menos que 3 dados para Shapiro-Wilk.")
        shapiro_df = pd.DataFrame(shapiro_results)
        if not shapiro_df.empty:
            st.dataframe(shapiro_df.set_index('Grupo'))
        else:
            st.info("Nenhum grupo com dados suficientes para teste de normalidade.")
        if not shapiro_df['p-valor'].isnull().all():
            if normality_by_group:
                st.success("Todos os grupos seguem distribuição normal (p >= 0.05).")
            else:
                st.warning("Algum grupo NÃO segue distribuição normal (p < 0.05).")
        else:
            st.info("Teste de normalidade não aplicável por dados insuficientes.")

    # Testes estatísticos principais
    st.markdown("#### Resultados do Teste Estatístico")
    cld_letters = {}
    num_groups = data_clean[group_var_name].nunique()

    if num_groups < 2:
        st.info("Número insuficiente de grupos para comparação.")
        return

    if homogeneous_variances and normality_by_group:
        st.info("Condições atendidas: usando **ANOVA paramétrica**.")
        with st.expander("Resultados da ANOVA"):
            try:
                formula = f'{dependent_var_name} ~ C({group_var_name})'
                model = ols(formula, data=data_clean).fit()
                anova_table = anova_lm(model, typ=2)
                st.dataframe(anova_table)
                if anova_table['PR(>F)'][0] < 0.05:
                    st.success("ANOVA SIGNIFICATIVA (p < 0.05).")
                    st.markdown("##### Teste post-hoc: Tukey HSD")
                    tukey_result = pairwise_tukeyhsd(endog=data_clean[dependent_var_name], groups=data_clean[group_var_name], alpha=0.05)
                    st.write(tukey_result)

                    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
                    p_matrix = pd.DataFrame(np.ones((num_groups, num_groups)), index=data_clean[group_var_name].unique(), columns=data_clean[group_var_name].unique())
                    for idx, row in tukey_df.iterrows():
                        p_matrix.loc[row['group1'], row['group2']] = row['p-adj']
                        p_matrix.loc[row['group2'], row['group1']] = row['p-adj']
                    cld_letters = get_compact_letter_display(p_matrix, data_clean[group_var_name].unique())
                    st.write("#### Letras de Significância (CLD):")
                    st.write(cld_letters)
                else:
                    st.info("ANOVA NÃO significativa (p >= 0.05).")
            except Exception as e:
                st.error(f"Erro na ANOVA: {e}")
    else:
        st.info("Condições não atendidas: usando **Kruskal-Wallis (não paramétrico)**.")
        with st.expander("Resultados do Kruskal-Wallis"):
            try:
                groups_kruskal = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() for g in data_clean[group_var_name].unique()]
                groups_kruskal = [g for g in groups_kruskal if len(g) > 0]
                if len(groups_kruskal) < 2:
                    st.info("Grupos insuficientes para Kruskal-Wallis.")
                else:
                    stat_kruskal, p_kruskal = stats.kruskal(*groups_kruskal)
                    st.write(f"Kruskal-Wallis H: {stat_kruskal:.3f}, p-valor: {p_kruskal:.3f}")
                    if p_kruskal < 0.05:
                        st.success("Kruskal-Wallis SIGNIFICATIVO (p < 0.05).")
                        st.markdown("##### Teste post-hoc: Dunn com correção Bonferroni")
                        dunn = sp.posthoc_dunn(data_clean, val_col=dependent_var_name, group_col=group_var_name, p_adjust='bonferroni')
                        st.dataframe(dunn)

                        cld_letters = get_compact_letter_display(dunn, data_clean[group_var_name].unique())
                        st.write("#### Letras de Significância (CLD):")
                        st.write(cld_letters)
                    else:
                        st.info("Kruskal-Wallis NÃO significativo (p >= 0.05).")
            except Exception as e:
                st.error(f"Erro no Kruskal-Wallis: {e}")

    # Gráfico boxplot com letras
    st.markdown("#### Boxplot com Letras de Significância")
    try:
        # Criar uma coluna com as letras CLD
        data_plot = data_clean.copy()
        data_plot['CLD'] = data_plot[group_var_name].map(cld_letters).fillna('')

        box = alt.Chart(data_plot).mark_boxplot(extent='min-max').encode(
            x=alt.X(f'{group_var_name}:N', title='Grupo'),
            y=alt.Y(f'{dependent_var_name}:Q', title=dependent_var_name.replace('_', ' ')),
            color=alt.Color(f'{group_var_name}:N', legend=None)
        )

        # Adiciona texto das letras de significância acima das caixas
        text = alt.Chart(data_plot.drop_duplicates(subset=[group_var_name])).mark_text(
            dy=-10,
            fontWeight='bold'
        ).encode(
            x=alt.X(f'{group_var_name}:N'),
            y=alt.Y(f'max({dependent_var_name}):Q'),
            detail=f'{group_var_name}:N',
            text=alt.Text('CLD:N')
        )

        st.altair_chart(box + text, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {e}")

# --- Interface Streamlit ---

st.title("Análise Estatística e Visualização - Vermicompostagem")
st.markdown("Carregue um arquivo Excel e selecione a variável para análise.")

# Upload do arquivo Excel
uploaded_file = st.file_uploader("Upload do arquivo Excel (.xlsx)", type=['xlsx'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Arquivo carregado com sucesso!")

        # Preencher os grupos manualmente conforme o seu código inicial:
        material_groups = {
            'Resíduo úmido': ['Resíduo úmido', 'RSU', 'Resíduo sólido urbano'],
            'Papel': ['Papel', 'Jornal', 'Cartão'],
            'Material lenhoso': ['Galhos', 'Folhas', 'Casca', 'Serragem'],
            'Matéria orgânica': ['Restos de comida', 'Comida', 'Orgânico'],
            'Outros': ['Plástico', 'Metal', 'Vidro']
        }

        # Criar coluna 'Grupo_material' no dataframe
        df['Grupo_material'] = df['Material'].map({m: g for g, lst in material_groups.items() for m in lst}).fillna('Outros')

        # Mostrar preview dos dados
        st.dataframe(df.head())

        # Escolher variável dependente
        variables = ['N_perc_final', 'P_perc_final', 'K_perc_final', 'Mg_perc_final', 'Ca_perc_final', 'pH_final', 'Temperatura_final']
        var_selected = st.selectbox("Selecione a variável para análise:", variables)

        run_statistical_analysis_and_plot(df, var_selected, 'Grupo_material')

    except Exception as e:
        st.error(f"Erro ao carregar ou processar o arquivo: {e}")
else:
    st.info("Por favor, faça o upload do arquivo Excel para começar.")
