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
def get_compact_letter_display(p_values_matrix, group_names):
    if not isinstance(p_values_matrix, pd.DataFrame):
        st.error("Internal Error: P-value matrix for CLD is not in the expected format (DataFrame).")
        return {group: '' for group in group_names}
    sorted_groups = sorted(group_names)
    significant_pairs = []
    for i in range(len(sorted_groups)):
        for j in range(i+1, len(sorted_groups)):
            g1, g2 = sorted_groups[i], sorted_groups[j]
            p_val = p_values_matrix.get(g1, {}).get(g2, p_values_matrix.get(g2, {}).get(g1, 1.0))
            if p_val < 0.05:
                significant_pairs.append(tuple(sorted((g1, g2))))
    clusters = []
    for g in sorted_groups:
        placed = False
        for cluster in clusters:
            if not any(tuple(sorted((g, m))) in significant_pairs for m in cluster):
                cluster.add(g); placed = True; break
        if not placed:
            clusters.append({g})
    cld = {g: ''.join(sorted(chr(ord('a') + i) for i, cluster in enumerate(clusters) if g in cluster))
           for g in group_names}
    return cld

def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    st.markdown(f"#### Analysis for: **{dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}**")
    data_clean = data.dropna(subset=[dependent_var_name, group_var_name])
    if data_clean[group_var_name].nunique() < 2:
        st.warning("Not enough groups."); return

    # Levene's Test
    with st.expander("Homogeneity of Variance (Levene)"):
        groups = [data_clean.loc[data_clean[group_var_name]==g, dependent_var_name] for g in data_clean[group_var_name].unique()]
        groups = [g for g in groups if len(g)>1]
        if len(groups) < 2:
            homogeneous_variances = False
            st.info("Não há grupos suficientes.")
        else:
            stat, p_levene = stats.levene(*groups)
            st.write(stat, p_levene)
            homogeneous_variances = p_levene >= 0.05
            st.success("Variâncias homogêneas" if homogeneous_variances else "Não homogêneas")

    # Shapiro-Wilk
    with st.expander("Normality Test (Shapiro-Wilk)"):
        normality = True
        for g in data_clean[group_var_name].unique():
            grp = data_clean.loc[data_clean[group_var_name]==g, dependent_var_name].dropna()
            if len(grp)>=3:
                _, p_s = stats.shapiro(grp)
                if p_s < 0.05: normality = False
        st.write("Normalidade por grupo:", normality)

    # Main test
    st.markdown("#### Statistical Test")
    num_groups = data_clean[group_var_name].nunique()
    cld_letters = {}
    if homogeneous_variances and normality:
        with st.expander("ANOVA"):
            mod = ols(f"{dependent_var_name} ~ C({group_var_name})", data=data_clean).fit()
            table = anova_lm(mod, typ=2)
            st.write(table)
            if table["PR(>F)"].iloc[0] < 0.05:
                res = pairwise_tukeyhsd(data_clean[dependent_var_name], data_clean[group_var_name])
                st.write(res)
                df_tuk = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
                mat = pd.DataFrame(np.ones((num_groups, num_groups)),
                                   index=data_clean[group_var_name].unique(),
                                   columns=data_clean[group_var_name].unique())
                for _, r in df_tuk.iterrows():
                    mat.loc[r["group1"], r["group2"]] = mat.loc[r["group2"], r["group1"]] = r["p-adj"]
                cld_letters = get_compact_letter_display(mat, data_clean[group_var_name].unique())
                st.write(cld_letters)
    else:
        with st.expander("Kruskal-Wallis"):
            grps = [data_clean.loc[data_clean[group_var_name]==g, dependent_var_name] for g in data_clean[group_var_name].unique()]
            h, p_kw = stats.kruskal(*[g for g in grps if len(g)>1])
            st.write(h, p_kw)
            if p_kw < 0.05:
                dunn = sp.posthoc_dunn(data_clean, dependent_var_name, group_var_name, p_adjust='bonferroni')
                st.dataframe(dunn)
                cld_letters = get_compact_letter_display(dunn, data_clean[group_var_name].unique())
                st.write(cld_letters)

    # Visualization
    st.markdown("#### Visualization")
    plot_df = data_clean.copy()
    plot_df["jitter_offset"] = np.random.uniform(-0.2, 0.2, len(plot_df))
    if cld_letters:
        plot_df["cld_letter"] = plot_df[group_var_name].map(cld_letters)
        maxvals = plot_df.groupby(group_var_name)[dependent_var_name].max().reset_index()
        buffer = plot_df[dependent_var_name].max() * 0.05
        maxvals["y_pos"] = maxvals[dependent_var_name] + buffer
        maxvals["cld_letter"] = maxvals[group_var_name].map(cld_letters)
    else:
        maxvals = pd.DataFrame()

    all_vals = plot_df[dependent_var_name].tolist()
    if not maxvals.empty:
        all_vals += maxvals["y_pos"].tolist()
    domain = [min(all_vals), max(all_vals)*1.1] if all_vals else [0, 1]

    base = alt.Chart(plot_df).encode(x=alt.X(f"{group_var_name}:N", axis=alt.Axis(labelAngle=-45)))
    box = base.mark_boxplot(size=50).encode(y=alt.Y(f"{dependent_var_name}:Q", scale=alt.Scale(domain=domain)))
    pts = base.mark_circle(opacity=0.6, size=60).encode(
        y=alt.Y(f"{dependent_var_name}:Q", scale=alt.Scale(domain=domain)),
        xOffset=alt.X("jitter_offset:Q"),
        tooltip=[group_var_name, dependent_var_name]
    )
    chart = box + pts
    if not maxvals.empty:
        lbls = alt.Chart(maxvals).mark_text(dy=-10).encode(
            x=alt.X(f"{group_var_name}:N"),
            y=alt.Y("y_pos:Q"),
            text="cld_letter:N"
        )
        chart = chart + lbls

    st.altair_chart(chart, use_container_width=True)

# --- Streamlit Config & Data Load ---
st.set_page_config(page_title="Vermicompost Meta‑analysis", layout="wide")
st.title("🔬 Vermicompost Meta‑analysis")
try:
    df = pd.read_excel("dados_vermicomposto.xlsx")
except FileNotFoundError:
    st.error("Arquivo não encontrado.")
    st.stop()

# Rename cols
col_map = {
    'Material de Origem do Vermicomposto': 'Source_Material',
    'N (%)': 'N_perc', 'P (%)': 'P_perc', 'K (%)': 'K_perc',
    'CN_Ratio_final': 'C_N_Ratio_final'
}
df.rename(columns={k: v for k, v in col_map.items() if k in df}, inplace=True)
required = list(col_map.values())
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Colunas faltando: {missing}")
    st.stop()

# Create groups
df["Material_Group"] = "Other"
patterns = {
    "Fruit & Vegetable Waste": r"Pineapple|Banana|Food Waste|Kitchen",
    "Coffee Waste": r"Coffee|SCG",
    "Manure & Related": r"Manure|Dung|Cow|Pig|Bagasse",
    "Diverse Plant Waste": r"Grass|Hyacinth|Parthenium|Bagasse",
    "Paper & Cellulose Waste": r"Newspaper|Paper|Cardboard"
}
for name, pat in patterns.items():
    df.loc[df["Source_Material"].str.contains(pat, case=False, na=False), "Material_Group"] = name
df["Material_Group"] = df["Material_Group"].astype("category")

# Variable selector
num_vars = {
    "Nitrogen (%)": "N_perc", "Phosphorus (%)": "P_perc",
    "Potassium (%)": "K_perc", "Final pH": "pH_final",
    "Final C/N Ratio": "C_N_Ratio_final"
}
sel = st.sidebar.selectbox("Select variable:", list(num_vars.keys()))
var = num_vars[sel]
run_statistical_analysis_and_plot(df, var, "Material_Group")
