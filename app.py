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
    sorted_groups = sorted(group_names)
    significant_pairs = []
    if not isinstance(p_values_matrix, pd.DataFrame):
        st.error("Internal Error: P-value matrix for CLD is not a DataFrame.")
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
                significant_pairs.append(tuple(sorted((g1, g2))))
    group_letters = {g: [] for g in group_names}
    clusters = []
    for g in sorted_groups:
        assigned_to_existing_cluster = False
        for cluster in clusters:
            if any(tuple(sorted((g, m))) in significant_pairs for m in cluster):
                continue
            cluster.add(g)
            assigned_to_existing_cluster = True
            break
        if not assigned_to_existing_cluster:
            clusters.append({g})
    for i, cluster in enumerate(clusters):
        letter = chr(ord('a') + i)
        for group in cluster:
            group_letters[group].append(letter)
    final_letters = {group: "".join(sorted(set(letters))) for group, letters in group_letters.items()}
    return final_letters

# --- Main Analysis Function ---
def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    st.markdown(f"#### Analysis for: **{dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}**")

    data_clean = data.dropna(subset=[dependent_var_name, group_var_name]).copy()

    if data_clean[group_var_name].nunique() < 2:
        st.warning(f"Not enough groups ({data_clean[group_var_name].nunique()}) for statistical analysis of {dependent_var_name}.")
        return

    with st.expander(f"Homogeneity of Variance Test (Levene's Test) for {dependent_var_name}"):
        groups = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() for g in data_clean[group_var_name].unique()]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            st.info("Not enough groups with data for Levene's Test.")
            homogeneous_variances = False
        else:
            stat, p_levene = stats.levene(*groups)
            st.write(f"Levene's Statistic: {stat:.3f}, p-value: {p_levene:.3f}")
            if p_levene < 0.05:
                st.warning("Variances are **NOT homogeneous** (p < 0.05).")
                homogeneous_variances = False
            else:
                st.success("Variances **ARE homogeneous** (p >= 0.05).")
                homogeneous_variances = True

    with st.expander(f"Normality Test (Shapiro-Wilk by group) for {dependent_var_name}"):
        normality_by_group = True
        shapiro_results = []
        for group in data_clean[group_var_name].unique():
            group_data = data_clean[data_clean[group_var_name] == group][dependent_var_name].dropna()
            if len(group_data) >= 3:
                stat_shapiro, p_shapiro = stats.shapiro(group_data)
                shapiro_results.append({'Group': group, 'N': len(group_data), 'Statistic': stat_shapiro, 'p-value': p_shapiro})
                if p_shapiro < 0.05:
                    normality_by_group = False
            else:
                shapiro_results.append({'Group': group, 'N': len(group_data), 'Statistic': np.nan, 'p-value': np.nan})
                st.info(f"Group '{group}' has less than 3 data points for Shapiro-Wilk Test. Skipped.")
        shapiro_df = pd.DataFrame(shapiro_results)
        st.dataframe(shapiro_df.set_index('Group'))
        if not shapiro_df['p-value'].isnull().all():
            if not normality_by_group:
                st.warning("At least one group **DOES NOT follow a normal distribution** (p < 0.05).")
            else:
                st.success("All tested groups **follow a normal distribution** (p >= 0.05).")
        else:
            st.info("Normality could not be tested due to small sample sizes.")

    st.markdown("#### Statistical Test Results")
    cld_letters = {}
    num_groups = data_clean[group_var_name].nunique()
    if num_groups < 2:
        st.info("Only one or no group found. Statistical tests not applicable.")
        return

    if homogeneous_variances and normality_by_group:
        st.info("Conditions met: Using **parametric ANOVA**.")
        with st.expander("ANOVA Results"):
            try:
                formula = f'{dependent_var_name} ~ C({group_var_name})'
                model = ols(formula, data=data_clean).fit()
                anova_table = anova_lm(model, typ=2)
                st.dataframe(anova_table)
                if anova_table['PR(>F)'].iloc[0] < 0.05:
                    st.success("ANOVA is **SIGNIFICANT** (p < 0.05).")
                    st.markdown("##### Post-hoc Test: Tukey HSD")
                    tukey_result = pairwise_tukeyhsd(endog=data_clean[dependent_var_name], groups=data_clean[group_var_name], alpha=0.05)
                    st.write(tukey_result)
                    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
                    p_matrix = pd.DataFrame(np.ones((num_groups, num_groups)), index=data_clean[group_var_name].unique(), columns=data_clean[group_var_name].unique())
                    for _, row in tukey_df.iterrows():
                        g1, g2, p_adj = row['group1'], row['group2'], row['p-adj']
                        p_matrix.loc[g1, g2] = p_adj
                        p_matrix.loc[g2, g1] = p_adj
                    cld_letters = get_compact_letter_display(p_matrix, data_clean[group_var_name].unique())
                    st.write("#### Significance Letters (CLD):")
                    st.write(cld_letters)
                else:
                    st.info("ANOVA is **NOT significant** (p >= 0.05). No differences detected.")
            except Exception as e:
                st.error(f"Error performing ANOVA: {e}")
    else:
        st.info("Conditions violated: Using **Kruskal-Wallis Test** (non-parametric).")
        with st.expander("Kruskal-Wallis Results"):
            try:
                groups_kruskal = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() for g in data_clean[group_var_name].unique()]
                groups_kruskal = [g for g in groups_kruskal if len(g) > 0]
                if len(groups_kruskal) < 2:
                    st.info("Not enough groups for Kruskal-Wallis Test.")
                else:
                    stat_kruskal, p_kruskal = stats.kruskal(*groups_kruskal)
                    st.write(f"Kruskal-Wallis H Statistic: {stat_kruskal:.3f}, p-value: {p_kruskal:.3f}")
                    if p_kruskal < 0.05:
                        st.success("Kruskal-Wallis test is **SIGNIFICANT** (p < 0.05).")
                        st.markdown("##### Post-hoc Test: Dunn with Bonferroni correction")
                        dunn_result = sp.posthoc_dunn(data_clean, val_col=dependent_var_name, group_col=group_var_name, p_adjust='bonferroni')
                        st.dataframe(dunn_result)
                        cld_letters = get_compact_letter_display(dunn_result, data_clean[group_var_name].unique())
                        st.write("#### Significance Letters (CLD):")
                        st.write(cld_letters)
                    else:
                        st.info("Kruskal-Wallis test is **NOT significant** (p >= 0.05). No differences detected.")
            except Exception as e:
                st.error(f"Error performing Kruskal-Wallis: {e}")

    # --- Visualization ---
    st.markdown("#### Data Visualization (Boxplot with Jitter)")

    plot_df = data_clean.copy()
    plot_df['jitter_offset'] = np.random.uniform(-0.2, 0.2, len(plot_df))

    base = alt.Chart(plot_df).encode(
        x=alt.X(f'{group_var_name}:N', title=group_var_name.replace('_', ' ')),
        y=alt.Y(f'{dependent_var_name}:Q', title=dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')),
        color=alt.Color(f'{group_var_name}:N', legend=None)
    )

    boxplot = base.mark_boxplot(size=20, extent='min-max').encode()

    points = base.mark_circle(size=40, opacity=0.5).encode(
        x=alt.X(f'{group_var_name}:N', axis=None),
        xOffset=alt.X('jitter_offset:Q', axis=None)
    )

    letter_df = pd.DataFrame([
        {'group': k, 'letter': v} for k, v in cld_letters.items()
    ])

    # Calculate y max per group to place letters above
    y_max = plot_df.groupby(group_var_name)[dependent_var_name].max().reset_index()
    letter_df = letter_df.merge(y_max, left_on='group', right_on=group_var_name)

    text = alt.Chart(letter_df).mark_text(
        dy=-10,
        fontWeight='bold',
        color='black'
    ).encode(
        x=alt.X(f'{group_var_name}:N', title=''),
        y=alt.Y(f'{dependent_var_name}:Q'),
        text='letter:N'
    )

    chart = (boxplot + points + text).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# --- Streamlit App ---

st.title("Vermicompost Analysis")

# Dummy Data Load / Replace with your own data source
@st.cache_data
def load_data():
    # You can replace this with your actual CSV file or other data source
    df = pd.DataFrame({
        'Material_Group': np.random.choice(['Group A', 'Group B', 'Group C'], 150),
        'N_perc': np.random.normal(2, 0.5, 150),
        'P_perc': np.random.normal(0.5, 0.1, 150),
        'K_perc': np.random.normal(1.2, 0.3, 150),
        'C_N_Ratio_final': np.random.normal(15, 4, 150),
        'pH': np.random.normal(7, 0.5, 150)
    })
    return df

df = load_data()

dependent_vars = ['N_perc', 'P_perc', 'K_perc', 'C_N_Ratio_final', 'pH']

selected_var = st.sidebar.selectbox("Select Variable for Analysis", dependent_vars)
group_var = 'Material_Group'

run_statistical_analysis_and_plot(df, selected_var, group_var)
