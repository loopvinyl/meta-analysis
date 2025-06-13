import pandas as pd
import streamlit as st
import altair as alt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np

# --- Helper Function for Compact Letter Display (CLD) ---
# This function assigns letters to groups based on their significance differences.
# Groups sharing a letter are not significantly different.
def get_compact_letter_display(p_values_matrix, group_names):
    # Based on a common algorithm for compact letter display
    # (e.g., similar to multcompView in R)
    
    # Initialize all groups with 'a'
    letters = {group: 'a' for group in group_names}
    
    # Sort groups to ensure consistent assignment
    sorted_groups = sorted(group_names)
    
    # Create a list of tuples (group1, group2, p_value) for significant differences
    significant_pairs = []
    for i in range(len(sorted_groups)):
        for j in range(i + 1, len(sorted_groups)):
            g1 = sorted_groups[i]
            g2 = sorted_groups[j]
            
            # Extract p-value from the symmetric matrix (either [g1,g2] or [g2,g1])
            p_val = p_values_matrix.loc[g1, g2] if g1 in p_values_matrix.index and g2 in p_values_matrix.columns else p_values_matrix.loc[g2, g1]
            
            if p_val < 0.05: # Assuming alpha = 0.05 for significance
                significant_pairs.append(tuple(sorted(tuple([g1, g2])))) # Store as sorted tuple

    # Iterate and assign letters
    current_letter = ord('a')
    letter_assignments = {} # {group: [letters it belongs to]}
    
    # Initialize all with an empty set of letters
    for group in sorted_groups:
        letter_assignments[group] = set()

    for group in sorted_groups:
        assigned = False
        for letter_set in letter_assignments.values():
            if chr(current_letter) in letter_set and not assigned:
                # Check for conflicts with current letter
                conflict = False
                for other_group in sorted_groups:
                    if chr(current_letter) in letter_assignments[other_group] and group != other_group:
                        # If two groups share a letter, they must not be significantly different
                        if tuple(sorted(tuple([group, other_group]))) in significant_pairs:
                            conflict = True
                            break
                if not conflict:
                    letter_assignments[group].add(chr(current_letter))
                    assigned = True
                    break
        
        if not assigned: # Assign a new letter if current conflicts or no letter assigned yet
            current_letter += 1
            letter_assignments[group].add(chr(current_letter))

    # Convert sets of letters to sorted strings
    final_letters = {group: "".join(sorted(list(letters))) for group, letters in letter_assignments.items()}
    return final_letters


# --- Function to run statistical analysis and display results ---
def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    st.markdown(f"#### Análise para: **{dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}**")

    # Clean data (remove NA for this specific analysis)
    data_clean = data.dropna(subset=[dependent_var_name, group_var_name]).copy()

    # Check if there's enough data for analysis
    if data_clean[group_var_name].nunique() < 2:
        st.warning(f"Não há grupos suficientes ({data_clean[group_var_name].nunique()}) para análise estatística de {dependent_var_name}.")
        return

    # 1. Teste de Homogeneidade de Variâncias (Levene's Test)
    with st.expander(f"Teste de Homogeneidade de Variâncias (Levene's Test) para {dependent_var_name}"):
        st.write("Avalia se as variâncias dos grupos são iguais.")
        try:
            # Statsmodels (recommended for formula interface) or scipy
            formula = f'{dependent_var_name} ~ C({group_var_name})'
            model = ols(formula, data=data_clean).fit()
            levene_result = anova_lm(model) # Levene test on residuals might be more robust
            # Using scipy.stats.levene directly on groups
            groups = [data_clean[dependent_var_name][data_clean[group_var_name] == g] for g in data_clean[group_var_name].unique()]
            stat, p_levene = stats.levene(*groups)
            st.write(f"Estatística de Levene: {stat:.3f}, p-valor: {p_levene:.3f}")
            if p_levene < 0.05:
                st.warning("Variâncias **NÃO são homogêneas** (p < 0.05). Isso pode sugerir o uso de testes não paramétricos ou correções.")
                homogeneous_variances = False
            else:
                st.success("Variâncias **SÃO homogêneas** (p >= 0.05).")
                homogeneous_variances = True
        except Exception as e:
            st.error(f"Não foi possível realizar o Teste de Levene: {e}")
            homogeneous_variances = False # Assume non-homogeneous if test fails


    # 2. Teste de Normalidade (Shapiro-Wilk por grupo)
    with st.expander(f"Teste de Normalidade (Shapiro-Wilk por grupo) para {dependent_var_name}"):
        st.write("Avalia se os dados em cada grupo seguem uma distribuição normal.")
        shapiro_results = []
        normality_by_group = True
        for group in data_clean[group_var_name].unique():
            group_data = data_clean[data_clean[group_var_name] == group][dependent_var_name].dropna()
            if len(group_data) >= 3: # Shapiro-Wilk requires at least 3 data points
                stat_shapiro, p_shapiro = stats.shapiro(group_data)
                shapiro_results.append({'Grupo': group, 'N': len(group_data), 'Estatística': stat_shapiro, 'p-valor': p_shapiro})
                if p_shapiro < 0.05:
                    normality_by_group = False
            else:
                shapiro_results.append({'Grupo': group, 'N': len(group_data), 'Estatística': np.nan, 'p-valor': np.nan})
                st.info(f"Grupo '{group}' tem menos de 3 pontos para Teste de Shapiro-Wilk. Não testado para normalidade.")

        shapiro_df = pd.DataFrame(shapiro_results)
        st.dataframe(shapiro_df.set_index('Grupo'))

        if not shapiro_df['p-valor'].isnull().all(): # Check if any p-values were calculated
            if not normality_by_group:
                st.warning("Pelo menos um grupo **NÃO segue uma distribuição normal** (p < 0.05).")
            else:
                st.success("Todos os grupos testados **seguem uma distribuição normal** (p >= 0.05).")
        else:
            st.info("Normalidade não pode ser testada para nenhum grupo (N muito pequeno).")

    # 3. Seleção e Execução dos Testes Estatísticos
    st.markdown("#### Resultados dos Testes Estatísticos")
    post_hoc_results_df = None
    cld_letters = {}
    
    num_groups = data_clean[group_var_name].nunique()

    if num_groups < 2:
        st.info("Apenas um grupo ou nenhum grupo encontrado para comparação. Testes estatísticos não aplicáveis.")
        return # Exit the function if not enough groups

    if homogeneous_variances and normality_by_group:
        st.info("Condições atendidas: Utilizando **ANOVA paramétrica**.")
        with st.expander("Resultados da ANOVA"):
            try:
                formula = f'{dependent_var_name} ~ C({group_var_name})'
                model = ols(formula, data=data_clean).fit()
                anova_table = anova_lm(model, typ=2) # Type 2 ANOVA sum of squares
                st.write("Tabela ANOVA:")
                st.dataframe(anova_table)

                if anova_table['PR(>F)'].iloc[0] < 0.05: # P-value for the group effect
                    st.success(f"A ANOVA é **SIGNIFICATIVA** (p < 0.05), indicando diferença entre os grupos.")
                    
                    st.markdown("##### Teste Post-hoc: Tukey HSD")
                    # Tukey HSD Post-hoc
                    tukey_result = pairwise_tukeyhsd(endog=data_clean[dependent_var_name],
                                                     groups=data_clean[group_var_name],
                                                     alpha=0.05)
                    st.write(tukey_result)

                    # Prepare p-values for CLD
                    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
                    p_matrix = pd.DataFrame(np.ones((num_groups, num_groups)), index=data_clean[group_var_name].unique(), columns=data_clean[group_var_name].unique())
                    for idx, row in tukey_df.iterrows():
                        group1 = row['group1']
                        group2 = row['group2']
                        p_adj = row['p-adj']
                        p_matrix.loc[group1, group2] = p_adj
                        p_matrix.loc[group2, group1] = p_adj # Symmetric
                    
                    cld_letters = get_compact_letter_display(p_matrix, data_clean[group_var_name].unique())
                    st.write("#### Letras de Significância (CLD):")
                    st.write(cld_letters)
                else:
                    st.info(f"A ANOVA **NÃO é significativa** (p >= 0.05). Não há diferença estatística detectada entre os grupos.")

            except Exception as e:
                st.error(f"Erro ao executar ANOVA: {e}")

    else:
        st.info("Condições VIOLADAS: Utilizando **Teste de Kruskal-Wallis** (não paramétrico).")
        with st.expander("Resultados do Kruskal-Wallis"):
            try:
                # Kruskal-Wallis Test
                groups_kruskal = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() for g in data_clean[group_var_name].unique()]
                stat_kruskal, p_kruskal = stats.kruskal(*groups_kruskal)
                st.write(f"Estatística H de Kruskal-Wallis: {stat_kruskal:.3f}, p-valor: {p_kruskal:.3f}")

                if p_kruskal < 0.05:
                    st.success(f"O teste de Kruskal-Wallis é **SIGNIFICATIVO** (p < 0.05), indicando diferença entre os grupos.")
                    
                    st.markdown("##### Teste Post-hoc: Dunn com correção de Bonferroni")
                    # Dunn's Post-hoc Test
                    dunn_result = sp.posthoc_dunn(data_clean, val_col=dependent_var_name,
                                                   group_col=group_var_name, p_adjust='bonferroni')
                    st.dataframe(dunn_result)

                    # Prepare p-values for CLD (Dunn's results are already a matrix)
                    cld_letters = get_compact_letter_display(dunn_result, data_clean[group_var_name].unique())
                    st.write("#### Letras de Significância (CLD):")
                    st.write(cld_letters)
                    
                else:
                    st.info(f"O teste de Kruskal-Wallis **NÃO é significativo** (p >= 0.05). Não há diferença estatística detectada entre os grupos.")
            except Exception as e:
                st.error(f"Erro ao executar Kruskal-Wallis: {e}")

    # --- Visualização (Boxplot com Jitter e CLD) ---
    st.markdown("#### Visualização dos Dados (Boxplot com Jitter)")
    
    # Adiciona as letras CLD ao dataframe para plotting
    plot_df = data_clean.copy()
    if cld_letters: # Only add if CLD was calculated
        plot_df['cld_letter'] = plot_df[group_var_name].map(cld_letters)
        # Calcula a posição Y para as letras (um pouco acima do valor máximo)
        plot_df_max_y = plot_df.groupby(group_var_name)[dependent_var_name].max().reset_index()
        plot_df_max_y['y_pos'] = plot_df_max_y[dependent_var_name] * 1.05 # 5% acima do max
        plot_df_max_y = plot_df_max_y.merge(plot_df[[group_var_name, 'cld_letter']].drop_duplicates(), on=group_var_name)
    
    base_chart = alt.Chart(plot_df).encode(
        x=alt.X(f"{group_var_name}:N", title="Grupo de Material", axis=alt.Axis(labelAngle=-45))
    ).properties(
        title=f"Distribuição de {dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')} por Grupo de Material"
    )

    # Boxplot layer
    boxplot = base_chart.mark_boxplot(size=60).encode(
        y=alt.Y(f"{dependent_var_name}:Q", title=dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')),
        color=alt.Color(f"{group_var_name}:N", legend=None) # Color by group, no legend needed
    )

    # Jitter points layer
    jitter = base_chart.mark_circle(size=80, opacity=0.7).encode(
        y=alt.Y(f"{dependent_var_name}:Q"),
        color=alt.Color(f"{group_var_name}:N", legend=None),
        xOffset=alt.Offset("jitter", band=0.5), # Add jitter
        tooltip=[group_var_name, dependent_var_name]
    )

    if cld_letters:
        # Text layer for CLD letters
        text_labels = alt.Chart(plot_df_max_y).mark_text(
            align='center',
            baseline='middle',
            dy=-10 # Adjust vertical position
        ).encode(
            x=alt.X(f"{group_var_name}:N"),
            y=alt.Y("y_pos:Q"),
            text=alt.Text("cld_letter:N"),
            color=alt.value("black")
        )
        chart = boxplot + jitter + text_labels
    else:
        chart = boxplot + jitter

    st.altair_chart(chart, use_container_width=True)

    # --- Interpretação dos Resultados ---
    with st.expander(f"✨ Interpretação Detalhada para {dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}"):
        st.markdown("A interpretação dos resultados deve ser feita considerando os testes de homogeneidade de variância e normalidade, e então os resultados da ANOVA ou Kruskal-Wallis e seus respectivos testes post-hoc.")
        
        st.markdown("##### Homogeneidade de Variâncias (Levene's Test):")
        if homogeneous_variances:
            st.markdown("- **P > 0.05**: As variâncias entre os grupos são consideradas **iguais** (homogêneas). Isso é bom para a ANOVA.")
        else:
            st.markdown("- **P < 0.05**: As variâncias entre os grupos são consideradas **diferentes** (não homogêneas). Isso sugere que a ANOVA pode não ser o teste mais apropriado, e o Kruskal-Wallis (não paramétrico) é uma alternativa robusta.")
        
        st.markdown("##### Normalidade (Shapiro-Wilk por Grupo):")
        if normality_by_group:
            st.markdown("- **P > 0.05 para todos os grupos**: Os dados em cada grupo seguem uma distribuição **normal**. Isso é uma premissa da ANOVA.")
        else:
            st.markdown("- **P < 0.05 para um ou mais grupos**: Os dados em pelo menos um grupo **não seguem uma distribuição normal**. Isso também aponta para o uso de testes não paramétricos como o Kruskal-Wallis.")
            
        st.markdown("##### Teste Principal (ANOVA ou Kruskal-Wallis):")
        if homogeneous_variances and normality_by_group:
            st.markdown("- **Se ANOVA P < 0.05**: Há uma diferença estatisticamente significativa entre as médias dos grupos para a variável analisada. Prossiga para o Tukey HSD para saber quais grupos são diferentes.")
            st.markdown("- **Se ANOVA P >= 0.05**: Não há evidência de diferença estatística entre as médias dos grupos.")
        else:
            st.markdown("- **Se Kruskal-Wallis P < 0.05**: Há uma diferença estatisticamente significativa entre as medianas (ou distribuições) dos grupos para a variável analisada. Prossiga para o Teste de Dunn para saber quais grupos são diferentes.")
            st.markdown("- **Se Kruskal-Wallis P >= 0.05**: Não há evidência de diferença estatística entre as medianas/distribuições dos grupos.")

        st.markdown("##### Teste Post-hoc (Tukey HSD ou Dunn):")
        if cld_letters:
            st.markdown("As **letras de significância (CLD)** no gráfico indicam os agrupamentos de forma compacta:")
            st.markdown("- Grupos que **compartilham a mesma letra** não são estatisticamente diferentes entre si.")
            st.markdown("- Grupos que **NÃO compartilham nenhuma letra** são estatisticamente diferentes entre si.")
            st.markdown("Por exemplo, se os grupos têm letras 'a', 'ab', 'b':")
            st.markdown("  - 'a' é diferente de 'b'.")
            st.markdown("  - 'ab' não é diferente de 'a' e não é diferente de 'b'.")
        else:
            st.markdown("Não houve teste post-hoc, pois o teste principal (ANOVA ou Kruskal-Wallis) não foi significativo, ou não houve grupos suficientes para comparação.")

# --- 2. Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Metanálise de Vermicompostos",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Análise Metanalítica de Vermicompostos por Tipo de Resíduo")
st.markdown("Este aplicativo interativo permite explorar os resultados dos testes estatísticos de Nitrogênio, Fósforo, Potássio, pH e Razão C/N em vermicompostos, agrupados por material de origem.")
st.markdown("---")

# --- 1. Carregamento de Dados ---
# O arquivo dados_vermicomposto.xlsx deve estar na mesma pasta que este script.
try:
    df = pd.read_excel('dados_vermicomposto.xlsx')
except FileNotFoundError:
    st.error("Erro: Arquivo 'dados_vermicomposto.xlsx' não encontrado. Por favor, certifique-se de que ele está na mesma pasta que este script.")
    st.stop()

# NOVO CÓDIGO AQUI: Criação da coluna Material_Group baseada na lógica do R
# Renomear 'Material de Origem do Vermicomposto' para 'Source_Material' temporariamente para a lógica.
if 'Material de Origem do Vermicomposto' in df.columns:
    df.rename(columns={'Material de Origem do Vermicomposto': 'Source_Material'}, inplace=True)
else:
    st.error("Erro: Coluna 'Material de Origem do Vermicomposto' não encontrada no arquivo de dados. Por favor, verifique o nome da coluna no seu Excel.")
    st.stop()

# Categorização Material_Group (como no código R)
df['Material_Group'] = "Uncategorized" # Default
df.loc[df['Source_Material'].str.contains("Pineapple|Abacaxi|Banana Leaf|Food Waste|Kitchen Waste", case=False, na=False), 'Material_Group'] = "Fruit & Vegetable Waste"
df.loc[df['Source_Material'].str.contains("Coffee|SCG", case=False, na=False), 'Material_Group'] = "Coffee Waste"
df.loc[df['Source_Material'].str.contains("Manure|Dung|Cattle Manure|Cow Manure|Pig Manure|Bagasse:|B0|B25|B50|B75", case=False, na=False), 'Material_Group'] = "Manure & Related"
df.loc[df['Source_Material'].str.contains("Grass Clippings|Water Hyacinth|Parthenium|Bagasse \\(100:0\\)|Pure Vermicompost|Matka khad|Kitchen-Yard Waste", case=False, na=False), 'Material_Group'] = "Diverse Plant Waste"
df.loc[df['Source_Material'].str.contains("Newspaper|Paper Waste|Cardboard", case=False, na=False), 'Material_Group'] = "Paper & Cellulose Waste"

df['Material_Group'] = df['Material_Group'].astype('category')


# Variáveis para análise
variaveis_numericas = {
    "Nitrogênio (%)": "N_perc",
    "Fósforo (%)": "P_perc",
    "Potássio (%)": "K_perc",
    "pH Final": "pH_final",
    "Razão C/N Final": "C_N_Ratio_final"
}

st.sidebar.header("Seleção da Variável para Análise")
variavel_selecionada = st.sidebar.selectbox(
    "Escolha a variável numérica para analisar:",
    list(variaveis_numericas.keys())
)

var_name = variaveis_numericas[variavel_selecionada]
group_col = "Material_Group" # O código agora cria e usa esta coluna

# --- Executa a Análise e Plota ---
run_statistical_analysis_and_plot(df, var_name, group_col)
