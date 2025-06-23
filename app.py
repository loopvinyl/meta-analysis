import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np
import string

# ------------------ CONFIGURAÇÕES DO APP ------------------
st.set_page_config(
    page_title="Vermicompost Meta-analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Meta-analysis of Vermicompost: Waste Type Impacts on Nutrients")
st.markdown("Explore statistical results for Nitrogen, Phosphorus, Potassium, pH, and C/N Ratio in vermicomposts")
st.markdown("---")

# ------------------ FUNÇÕES AUXILIARES ------------------

def assign_material_group(source):
    if pd.isna(source):
        return "Uncategorized"
    source = str(source).lower()
    
    manure_keywords = ["manure", "dung", "cattle", "cow", "bovine", "cd", "vr", "fezes", "estrume", "gado", "vaca"]
    if any(kw in source for kw in manure_keywords):
        return "Manure-Based"
    
    coffee_keywords = ["coffee", "scg", "borra", "café"]
    if any(kw in source for kw in coffee_keywords):
        return "Coffee Waste"
    
    agro_industrial_keywords = ["pineapple", "abacaxi", "fruit", "fruta", "peels", 
                                "food", "kitchen", "alimento", 
                                "bagasse", "crop", "residue", "resíduo", "straw", "palha", "sugarcane", "bagaço"]
    if any(kw in source for kw in agro_industrial_keywords):
        return "Agro-Industrial Waste"
    
    plant_keywords = ["vegetable", "grass", "water hyacinth", "weeds", "parthenium", "green", "verde", "hortaliças"]
    if any(kw in source for kw in plant_keywords):
        return "Plant Waste"
    
    cellulosic_keywords = ["cardboard", "paper", "filters", "filtro", "cellulose", "papel", "papelão"]
    if any(kw in source for kw in cellulosic_keywords):
        return "Cellulosic Waste"
    
    return "Uncategorized"

def get_category_description(category):
    descriptions = {
        "Manure-Based": "Todos os vermicompostos com base em esterco animal (puros ou misturas)",
        "Coffee Waste": "Resíduos de café processado (borra) sem esterco",
        "Agro-Industrial Waste": "Resíduos de processamento agrícola e industrial (frutas, alimentos, cultivos)",
        "Plant Waste": "Materiais vegetais frescos (hortaliças, grama, plantas aquáticas)",
        "Cellulosic Waste": "Materiais ricos em celulose (papelão, filtros, papel)"
    }
    return descriptions.get(category, "Sem descrição disponível")

def get_compact_letter_display(p_values_matrix, group_names):
    sorted_groups = sorted(group_names)
    significant_pairs = []

    if not isinstance(p_values_matrix, pd.DataFrame):
        st.error("Erro interno: matriz de p-valores inválida.")
        return {group: '' for group in group_names}
    
    for i in range(len(sorted_groups)):
        for j in range(i + 1, len(sorted_groups)):
            g1, g2 = sorted_groups[i], sorted_groups[j]
            try:
                p_val = p_values_matrix.loc[g1, g2]
            except KeyError:
                try:
                    p_val = p_values_matrix.loc[g2, g1]
                except KeyError:
                    p_val = 1.0
            if p_val < 0.05:
                significant_pairs.append(tuple(sorted((g1, g2))))

    group_letters = {g: [] for g in group_names}
    clusters = []

    for g in sorted_groups:
        assigned = False
        for cluster in clusters:
            if all(tuple(sorted((g, member))) not in significant_pairs for member in cluster):
                cluster.add(g)
                assigned = True
                break
        if not assigned:
            clusters.append({g})
    
    letters = list(string.ascii_lowercase)
    for i, cluster in enumerate(clusters):
        letter = letters[i] if i < len(letters) else str(i)
        for group in cluster:
            group_letters[group].append(letter)

    final = {group: ''.join(sorted(set(letters))) for group, letters in group_letters.items()}
    return final

# ------------------ CARREGAR DADOS ------------------
try:
    df = pd.read_excel('dados_vermicomposto_v6.xlsx', sheet_name='Planilha1')

    column_map = {
        'Article (Authors, Year)': 'Article',
        'Source_Material': 'Source_Material',
        'N (%)': 'N_perc',
        'Ref. N (%)': 'Ref_N_perc',
        'P (%)': 'P_perc',
        'Ref. P (%)': 'Ref_P_perc',
        'K (%)': 'K_perc',
        'Ref. K (%)': 'Ref_K_perc',
        'pH_final': 'pH_final',
        'Ref. pH_final': 'Ref_pH_final',
        'CN_Ratio_final': 'C_N_Ratio_final',
        'Ref. CN_Ratio_final': 'Ref_CN_Ratio',
        'Duration_days': 'Duration_days',
        'Ref. Duration_days': 'Ref_Duration_days',
        'Additional Observations': 'Observations'
    }
    df.rename(columns=column_map, inplace=True)

    numeric_cols = ['N_perc', 'P_perc', 'K_perc', 'pH_final', 'C_N_Ratio_final', 'Duration_days']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

except Exception as e:
    st.error(f"Erro ao carregar dados: {str(e)}")
    st.stop()

# ------------------ CATEGORIZAÇÃO ------------------
df['Material_Group'] = df['Source_Material'].apply(assign_material_group)

# ------------------ VARIÁVEIS ------------------
numerical_variables = {
    "Nitrogen (%)": "N_perc",
    "Phosphorus (%)": "P_perc",
    "Potassium (%)": "K_perc",
    "Final pH": "pH_final",
    "Final C/N Ratio": "C_N_Ratio_final"
}

# ------------------ INTERFACE ------------------
st.sidebar.header("Parâmetros de Análise")
selected_var = st.sidebar.selectbox("Selecione o parâmetro:", list(numerical_variables.keys()))
selected_col = numerical_variables[selected_var]

# ------------------ ANÁLISE ESTATÍSTICA ------------------

def run_analysis(data, y_var, group_var):
    st.header(f"Análise de {y_var.replace('_', ' ')}")
    data = data.dropna(subset=[y_var, group_var])
    if data[group_var].nunique() < 2:
        st.warning("Menos de dois grupos presentes.")
        return

    # Homogeneidade
    groups = [grp[y_var].dropna() for _, grp in data.groupby(group_var)]
    stat_levene, p_levene = stats.levene(*groups)
    st.write(f"Levene: p={p_levene:.4f}")
    homogeneous = p_levene >= 0.05

    # Normalidade
    normal = True
    for g in groups:
        if len(g) >= 3:
            _, p = stats.shapiro(g)
            if p < 0.05:
                normal = False

    if homogeneous and normal:
        st.subheader("ANOVA")
        model = ols(f'{y_var} ~ C({group_var})', data=data).fit()
        anova_table = anova_lm(model)
        st.dataframe(anova_table)

        if anova_table['PR(>F)'].iloc[0] < 0.05:
            tukey = pairwise_tukeyhsd(endog=data[y_var], groups=data[group_var], alpha=0.05)
            st.text(str(tukey))
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            groups_unique = data[group_var].unique()
            p_matrix = pd.DataFrame(np.ones((len(groups_unique), len(groups_unique))), index=groups_unique, columns=groups_unique)
            for _, row in tukey_df.iterrows():
                p_matrix.loc[row['group1'], row['group2']] = row['p-adj']
                p_matrix.loc[row['group2'], row['group1']] = row['p-adj']
            cld = get_compact_letter_display(p_matrix, groups_unique)
            st.write("Letras de significância:", cld)
    else:
        st.subheader("Kruskal-Wallis")
        stat, p = stats.kruskal(*groups)
        st.write(f"Kruskal-Wallis: p={p:.4f}")
        if p < 0.05:
            result = sp.posthoc_dunn(data, val_col=y_var, group_col=group_var, p_adjust='bonferroni')
            st.dataframe(result)
            cld = get_compact_letter_display(result, data[group_var].unique())
            st.write("Letras de significância:", cld)

    # ------------------ VISUALIZAÇÃO ------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=group_var, y=y_var, ax=ax)
    sns.stripplot(data=data, x=group_var, y=y_var, color='black', alpha=0.5, jitter=True, ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

run_analysis(df, selected_col, "Material_Group")
