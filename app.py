import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import io

# ----------------------------
# CONFIGURAÇÃO STREAMLIT
# ----------------------------
st.set_page_config(page_title="Vermicomposto", layout="wide")
st.title("🔬 Meta-análise de Vermicomposto: Impactos do Tipo de Resíduo")

# ----------------------------
# DADOS CSV EMBUTIDOS
# ----------------------------
dados_csv = """
Article,Source_Material,N_perc,P_perc,K_perc,pH_final,C_N_Ratio_final,Duration_days
Adi & Noor (2009),Bovine Manure:Kitchen Waste (30:70) - T1,1.07,0.32,0.41,,14.1,70
Adi & Noor (2009),Bovine Manure:Spent Coffee Grounds (30:70) - T2,1.06,0.32,0.41,,14.2,70
Aliyat et al. (2000),Food Waste Vermicompost,1.8,0.5,0.7,7.3,12.0,90
Bhat et al. (2015),Bagasse:Cattle Dung (0:100) - B0,1.3,0.46,0.47,7.32,18.92,135
Bhat et al. (2015),Bagasse:Cattle Dung (25:75) - B25,1.28,0.43,0.49,7.41,16.83,135
Bhat et al. (2015),Bagasse:Cattle Dung (50:50) - B50,1.16,0.4,0.51,7.3,16.72,135
Bhat et al. (2015),Bagasse:Cattle Dung (75:25) - B75,0.9,0.35,0.44,7.13,20.22,135
Bhat et al. (2015),Pure Bagasse (100:0) - B100,0.3,0.2,0.36,7.3,57.59,135
Goswami et al. (2017),Vegetable waste, crop residue, cow dung,1.0,0.25,0.3,7.3,6.05,30
Mago et al. (2021),100% Cow Dung (V0),0.98,0.31,0.29,7.1,8.90,105
Mago et al. (2021),60% CD + 40% BL (V2),0.96,0.33,0.31,7.1,8.30,105
Mago et al. (2021),40% CD + 60% BL (V3),0.93,0.35,0.32,7.1,9.90,105
Zziwa et al. (2021),Pineapple peels + Cattle manure (4:1),0.63,0.4,0.3,6.9,55.86,72
Zziwa et al. (2021),Pineapple peels + Cattle manure (4:1),0.62,0.38,0.31,6.0,52.33,72
Liu and Price (2011),Spent Coffee Grounds (SCG) + Cardboard,1.1,0.28,0.34,7.2,19.5,72
Liu and Price (2011),SCG + Coffee Filters,1.05,0.27,0.3,7.2,21.1,72
Demerew and Abera (2024),Avocado waste + Cow dung,1.02,0.3,0.4,7.3,12.0,60
Demerew and Abera (2024),Coffee husk + Avocado + Cow dung,0.92,0.27,0.39,7.2,14.0,60
"""

# Lê o CSV direto do texto
df = pd.read_csv(io.StringIO(dados_csv))

# ----------------------------
# AGRUPAMENTO
# ----------------------------
def assign_material_group(source):
    if pd.isna(source):
        return "Uncategorized"
    source = str(source).lower()
    if any(kw in source for kw in ["manure", "dung", "cow", "bovine", "gado", "vaca", "estrume", "fezes"]):
        return "Manure-Based"
    if any(kw in source for kw in ["coffee", "scg", "borra", "café", "coffee husk"]):
        return "Coffee Waste"
    if any(kw in source for kw in ["fruit", "fruta", "peels", "food", "sugarcane", "bagasse", "residue", "waste", "straw"]):
        return "Agro-Industrial Waste"
    if any(kw in source for kw in ["vegetable", "grass", "verde", "hortaliça"]):
        return "Plant Waste"
    if any(kw in source for kw in ["cardboard", "paper", "filters", "celulose", "papel"]):
        return "Cellulosic Waste"
    return "Uncategorized"

df["Material_Group"] = df["Source_Material"].apply(assign_material_group)

# ----------------------------
# CONVERSÃO PARA NUMÉRICO
# ----------------------------
for col in ['N_perc', 'P_perc', 'K_perc', 'pH_final', 'C_N_Ratio_final', 'Duration_days']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ----------------------------
# INTERFACE
# ----------------------------
variaveis = {
    "Nitrogen (%)": "N_perc",
    "Phosphorus (%)": "P_perc",
    "Potassium (%)": "K_perc",
    "Final pH": "pH_final",
    "C/N Ratio": "C_N_Ratio_final"
}
st.sidebar.header("📌 Variável para análise")
var_escolhida = st.sidebar.selectbox("Escolha a variável:", list(variaveis.keys()))
coluna = variaveis[var_escolhida]

# ----------------------------
# ANÁLISE ESTATÍSTICA
# ----------------------------
st.subheader(f"📊 Análise Estatística: {var_escolhida}")
dados = df.dropna(subset=[coluna, 'Material_Group'])

grupos = [g[coluna] for _, g in dados.groupby("Material_Group")]
normal = True
homogenio = False

if len(grupos) >= 2:
    _, p_levene = stats.levene(*grupos)
    homogenio = p_levene >= 0.05
    st.write(f"Teste de Levene (homogeneidade): p = {p_levene:.4f}")

    for g in grupos:
        if len(g) >= 3:
            _, p_shapiro = stats.shapiro(g)
            if p_shapiro < 0.05:
                normal = False

    if normal and homogenio:
        st.success("✅ ANOVA aplicada")
        modelo = ols(f"{coluna} ~ C(Material_Group)", data=dados).fit()
        anova_result = anova_lm(modelo)
        st.dataframe(anova_result)

        if anova_result['PR(>F)'][0] < 0.05:
            tukey = pairwise_tukeyhsd(dados[coluna], dados['Material_Group'])
            st.text("Tukey HSD:")
            st.text(tukey)
    else:
        st.warning("⚠️ Assunções violadas. Usando Kruskal-Wallis")
        _, p_kruskal = stats.kruskal(*grupos)
        st.write(f"Kruskal-Wallis: p = {p_kruskal:.4f}")
        if p_kruskal < 0.05:
            dunn = sp.posthoc_dunn(dados, val_col=coluna, group_col="Material_Group", p_adjust="bonferroni")
            st.dataframe(dunn)
else:
    st.warning("⚠️ Dados insuficientes para análise.")

# ----------------------------
# VISUALIZAÇÃO
# ----------------------------
st.subheader("📈 Visualização Gráfica")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=dados, x="Material_Group", y=coluna, ax=ax)
sns.stripplot(data=dados, x="Material_Group", y=coluna, color="black", alpha=0.6, jitter=True, ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)
