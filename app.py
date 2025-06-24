import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np

# --- Dados embutidos diretamente (versão reduzida para exemplo) ---
data = {
    'Article': [
        'Adi & Noor (2009)', 'Adi & Noor (2009)', 'Aliyah et al. (2000)',
        'Bhat et al. (2015)', 'Zziwa et al. (2021)', 'Demerew and Abera (2024)'
    ],
    'Source_Material': [
        'Bovine Manure:Kitchen Waste', 
        'Bovine Manure:Spent Coffee Grounds',
        'Food Waste Vermicompost',
        'Bagasse:Cattle Dung (25:75)',
        'Pineapple peels + Cattle manure (4:1)',
        'Coffee husk + Cow dung'
    ],
    'N_perc': [1.07, 1.09, 0.80, 0.68, 0.41, 1.10],
    'P_perc': [0.32, 0.29, 0.10, 0.22, 0.07, 0.30],
    'K_perc': [0.41, 0.37, 0.09, 0.26, 0.08, 0.35],
    'pH_final': [7.0, 7.1, 7.3, 7.41, 6.3, 7.2],
    'C_N_Ratio_final': [14.1, 12.4, 22.0, 18.9, 25.8, 13.2],
    'Duration_days': [70, 70, 90, 135, 60, 60]
}

df = pd.DataFrame(data)

# --- Agrupamento de materiais ---
def assign_material_group(source):
    if pd.isna(source):
        return "Uncategorized"
    source = str(source).lower()
    if any(kw in source for kw in ["manure", "cow", "dung", "gado", "vaca", "estrume"]):
        return "Manure-Based"
    if any(kw in source for kw in ["coffee", "scg", "borra", "café"]):
        return "Coffee Waste"
    if any(kw in source for kw in ["fruit", "peels", "sugarcane", "bagasse", "bagaço"]):
        return "Agro-Industrial Waste"
    if any(kw in source for kw in ["vegetable", "grass", "verde"]):
        return "Plant Waste"
    return "Uncategorized"

df['Material_Group'] = df['Source_Material'].apply(assign_material_group)

# --- Interface Streamlit ---
st.set_page_config(page_title="Vermicompost Analysis", layout="wide")
st.title("🌱 Meta-analysis of Vermicompost Quality")

variaveis = {
    "Nitrogen (%)": "N_perc",
    "Phosphorus (%)": "P_perc",
    "Potassium (%)": "K_perc",
    "Final pH": "pH_final",
    "C/N Ratio": "C_N_Ratio_final"
}

st.sidebar.header("🔍 Selecione a variável:")
var_escolhida = st.sidebar.selectbox("Variável a analisar:", list(variaveis.keys()))
coluna = variaveis[var_escolhida]

# --- Análise estatística ---
st.subheader(f"📊 Análise estatística: {var_escolhida}")
dados = df.dropna(subset=[coluna, 'Material_Group'])
grupos = [g[coluna].dropna() for _, g in dados.groupby("Material_Group")]

homogenio = False
normal = True

if len(grupos) >= 2:
    stat_levene, p_levene = stats.levene(*grupos)
    homogenio = p_levene >= 0.05
    st.write(f"Teste de Levene: p = {p_levene:.4f}")

    for g in grupos:
        if len(g) >= 3:
            _, p = stats.shapiro(g)
            if p < 0.05:
                normal = False

    if homogenio and normal:
        st.success("✅ Assunções atendidas: ANOVA")
        modelo = ols(f"{coluna} ~ C(Material_Group)", data=dados).fit()
        anova_result = anova_lm(modelo)
        st.dataframe(anova_result)

        if anova_result['PR(>F)'][0] < 0.05:
            st.info("Diferença significativa detectada (ANOVA).")
    else:
        st.warning("⚠️ Assunções violadas. Usando Kruskal-Wallis")
        stat, p = stats.kruskal(*grupos)
        st.write(f"Kruskal-Wallis: p = {p:.4f}")
        if p < 0.05:
            st.info("Diferença significativa detectada (Kruskal-Wallis).")
else:
    st.warning("Dados insuficientes para análise estatística.")

# --- Visualização ---
st.subheader("📈 Distribuição dos dados")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=dados, x="Material_Group", y=coluna, ax=ax)
sns.stripplot(data=dados, x="Material_Group", y=coluna, color="black", alpha=0.5, jitter=True, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)
