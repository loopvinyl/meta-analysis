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

# --- Função para categorizar o material ---
def assign_material_group(source):
    if pd.isna(source):
        return "Uncategorized"
    source = str(source).lower()
    
    if any(kw in source for kw in ["manure", "dung", "cattle", "cow", "bovine", "estrume", "gado", "vaca", "fezes"]):
        return "Manure-Based"
    if any(kw in source for kw in ["coffee", "scg", "borra", "café"]):
        return "Coffee Waste"
    if any(kw in source for kw in ["fruit", "fruta", "peels", "food", "alimento", "resíduo", "sugarcane", "bagaço", "straw", "bagasse"]):
        return "Agro-Industrial Waste"
    if any(kw in source for kw in ["vegetable", "grass", "verde", "hortaliças"]):
        return "Plant Waste"
    if any(kw in source for kw in ["cardboard", "paper", "filters", "cellulose", "papel"]):
        return "Cellulosic Waste"
    return "Uncategorized"

# --- Carregamento dos dados ---
st.set_page_config(page_title="Vermicompost Analysis", layout="wide")
st.title("🔬 Meta-analysis of Vermicompost: Waste Type Impacts")

try:
    df = pd.read_excel("dados_vermicomposto_v6.xlsx", sheet_name="Planilha1")
except Exception as e:
    st.error(f"Erro ao carregar a planilha: {e}")
    st.stop()

# --- Renomear colunas ---
colunas_esperadas = {
    'Article (Authors, Year)': 'Article',
    'Source_Material': 'Source_Material',
    'N (%)': 'N_perc',
    'P (%)': 'P_perc',
    'K (%)': 'K_perc',
    'pH_final': 'pH_final',
    'CN_Ratio_final': 'C_N_Ratio_final',
    'Duration_days': 'Duration_days'
}

df.rename(columns=colunas_esperadas, inplace=True)
df['Material_Group'] = df['Source_Material'].apply(assign_material_group)

# --- Converter para numérico ---
for col in ['N_perc', 'P_perc', 'K_perc', 'pH_final', 'C_N_Ratio_final', 'Duration_days']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Interface ---
variaveis = {
    "Nitrogen (%)": "N_perc",
    "Phosphorus (%)": "P_perc",
    "Potassium (%)": "K_perc",
    "Final pH": "pH_final",
    "C/N Ratio": "C_N_Ratio_final"
}

st.sidebar.header("Seleção")
var_escolhida = st.sidebar.selectbox("Variável a analisar:", list(variaveis.keys()))
coluna = variaveis[var_escolhida]

# --- Estatísticas ---
st.subheader(f"Análise estatística: {var_escolhida}")
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
        st.success("Condições atendidas: ANOVA")
        modelo = ols(f"{coluna} ~ C(Material_Group)", data=dados).fit()
        anova_result = anova_lm(modelo)
        st.dataframe(anova_result)

        if anova_result['PR(>F)'][0] < 0.05:
            tukey = pairwise_tukeyhsd(dados[coluna], dados['Material_Group'])
            st.text(str(tukey))
    else:
        st.warning("Condições não atendidas: Kruskal-Wallis")
        stat, p = stats.kruskal(*grupos)
        st.write(f"Kruskal-Wallis: p = {p:.4f}")
        if p < 0.05:
            dunn = sp.posthoc_dunn(dados, val_col=coluna, group_col="Material_Group", p_adjust="bonferroni")
            st.dataframe(dunn)
else:
    st.warning("Dados insuficientes para análise.")

# --- Visualização ---
st.subheader("Distribuição dos dados")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=dados, x="Material_Group", y=coluna, ax=ax)
sns.stripplot(data=dados, x="Material_Group", y=coluna, color="black", alpha=0.5, jitter=True, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)
