import pandas as pd
import streamlit as st
import altair as alt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np

# --- Improved Helper Function for Compact Letter Display (CLD) ---
def get_compact_letter_display(p_values_matrix, group_names):
    """Improved CLD function with better handling of non-significant groups"""
    sorted_groups = sorted(group_names)
    significant_pairs = []
    
    # Handle different matrix types
    if isinstance(p_values_matrix, pd.DataFrame):
        matrix = p_values_matrix
    else:
        matrix = pd.DataFrame(
            p_values_matrix, 
            index=group_names,
            columns=group_names
        )
    
    # Collect significant pairs
    for i, g1 in enumerate(sorted_groups):
        for j, g2 in enumerate(sorted_groups[i+1:], i+1):
            p_val = matrix.loc[g1, g2] if g1 in matrix.index and g2 in matrix.columns else 1.0
            if p_val < 0.05:
                significant_pairs.append((g1, g2))
    
    # Assign letters to groups
    groups_letters = {g: [] for g in sorted_groups}
    current_letter = ord('a')
    
    # Process groups until all are assigned
    while sorted_groups:
        group = sorted_groups.pop(0)
        if not groups_letters[group]:
            groups_letters[group].append(chr(current_letter))
            current_letter += 1
        
        # Find non-conflicting groups
        for other in sorted_groups[:]:
            if (group, other) not in significant_pairs and (other, group) not in significant_pairs:
                groups_letters[other] = groups_letters[group].copy()
                sorted_groups.remove(other)
    
    return {g: ''.join(letters) for g, letters in groups_letters.items()}

# --- Enhanced Main Analysis Function ---
def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    st.markdown(f"#### Analysis for: **{dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}**")

    # Data cleaning with informative messages
    initial_count = len(data)
    data_clean = data.dropna(subset=[dependent_var_name, group_var_name]).copy()
    removed_count = initial_count - len(data_clean)
    
    if removed_count > 0:
        st.info(f"Removed {removed_count} rows with missing values")
    
    unique_groups = data_clean[group_var_name].unique()
    num_groups = len(unique_groups)
    
    if num_groups < 2:
        st.warning(f"Not enough groups ({num_groups}) for statistical analysis")
        return

    # Homogeneity of Variance Test
    with st.expander("Homogeneity of Variance (Levene's Test)"):
        groups = [data_clean.loc[data_clean[group_var_name]==g, dependent_var_name] 
                 for g in unique_groups]
        
        if any(len(g) < 2 for g in groups):
            st.warning("Some groups have less than 2 observations - test skipped")
            homogeneous_variances = False
        else:
            try:
                stat, p_levene = stats.levene(*groups)
                st.write(f"Levene's Statistic: {stat:.4f}, p-value: {p_levene:.4f}")
                homogeneous_variances = p_levene >= 0.05
                st.success("Variances ARE homogeneous") if homogeneous_variances \
                    else st.warning("Variances are NOT homogeneous")
            except Exception as e:
                st.error(f"Levene's test failed: {str(e)}")
                homogeneous_variances = False

    # Normality Test
    with st.expander("Normality Test (Shapiro-Wilk)"):
        normality_results = []
        normality_by_group = True
        
        for group in unique_groups:
            group_data = data_clean.loc[data_clean[group_var_name]==group, dependent_var_name]
            n = len(group_data)
            
            if n < 3:
                result = {'Group': group, 'n': n, 'W': None, 'p-value': None, 
                          'Result': 'Insufficient data (n<3)'}
                normality_by_group = False
            else:
                try:
                    W, p = stats.shapiro(group_data)
                    result = {'Group': group, 'n': n, 'W': f"{W:.4f}", 
                              'p-value': f"{p:.4f}", 
                              'Result': 'Normal' if p >= 0.05 else 'Non-normal'}
                    if p < 0.05: 
                        normality_by_group = False
                except Exception as e:
                    result = {'Group': group, 'n': n, 'W': 'Error', 
                              'p-value': 'Error', 'Result': str(e)}
                    normality_by_group = False
            
            normality_results.append(result)
        
        st.dataframe(pd.DataFrame(normality_results))
        
        if normality_by_group:
            st.success("All groups meet normality assumption")
        else:
            st.warning("Normality assumption violated in at least one group")

    # Statistical Testing
    st.markdown("#### Statistical Test Results")
    cld_letters = {}
    
    if homogeneous_variances and normality_by_group:
        st.info("Using **parametric ANOVA** (conditions met)")
        try:
            model = ols(f'{dependent_var_name} ~ C({group_var_name})', data=data_clean).fit()
            anova_table = anova_lm(model, typ=2)
            
            with st.expander("ANOVA Results"):
                st.dataframe(anova_table.style.format("{:.4f}"))
                
                p_value = anova_table.loc[f'C({group_var_name})', 'PR(>F)']
                if p_value < 0.05:
                    st.success("Significant difference detected (p < 0.05)")
                    
                    # Post-hoc test
                    st.markdown("##### Tukey HSD Post-hoc Test")
                    tukey = pairwise_tukeyhsd(data_clean[dependent_var_name], 
                                              data_clean[group_var_name])
                    st.text(str(tukey))
                    
                    # Create p-value matrix for CLD
                    tukey_df = pd.DataFrame(tukey._results_table.data[1:], 
                                           columns=tukey._results_table.data[0])
                    p_matrix = pd.DataFrame(
                        1.0, 
                        index=unique_groups, 
                        columns=unique_groups
                    )
                    
                    for _, row in tukey_df.iterrows():
                        g1, g2 = row['group1'], row['group2']
                        p_matrix.loc[g1, g2] = row['p-adj']
                        p_matrix.loc[g2, g1] = row['p-adj']
                    
                    cld_letters = get_compact_letter_display(p_matrix, unique_groups)
                else:
                    st.info("No significant differences found")
        except Exception as e:
            st.error(f"ANOVA failed: {str(e)}")
    else:
        st.info("Using **Kruskal-Wallis** (non-parametric)")
        try:
            valid_groups = []
            group_data = []
            
            for group in unique_groups:
                g_data = data_clean.loc[data_clean[group_var_name]==group, dependent_var_name]
                if len(g_data) > 0:
                    valid_groups.append(group)
                    group_data.append(g_data)
            
            if len(valid_groups) < 2:
                st.warning("Insufficient groups with data")
                return
                
            H, p_kruskal = stats.kruskal(*group_data)
            
            with st.expander("Kruskal-Wallis Results"):
                st.write(f"H-statistic: {H:.4f}, p-value: {p_kruskal:.4f}")
                
                if p_kruskal < 0.05:
                    st.success("Significant difference detected (p < 0.05)")
                    
                    # Post-hoc test
                    st.markdown("##### Dunn's Post-hoc Test with Bonferroni correction")
                    dunn_result = sp.posthoc_dunn(
                        data_clean, 
                        val_col=dependent_var_name, 
                        group_col=group_var_name, 
                        p_adjust='bonferroni'
                    )
                    st.dataframe(dunn_result.style.format("{:.4f}"))
                    
                    cld_letters = get_compact_letter_display(dunn_result, valid_groups)
                else:
                    st.info("No significant differences found")
        except Exception as e:
            st.error(f"Kruskal-Wallis failed: {str(e)}")

    # Enhanced Visualization
    st.markdown("#### Visualization")
    
    if not cld_letters:
        cld_letters = {g: "" for g in unique_groups}
    
    plot_df = data_clean.copy()
    plot_df['jitter'] = np.random.uniform(-0.2, 0.2, size=len(plot_df))
    
    # Calculate positions for CLD letters
    y_max = plot_df.groupby(group_var_name)[dependent_var_name].max()
    y_span = plot_df[dependent_var_name].max() - plot_df[dependent_var_name].min()
    letter_y = y_max + (0.05 * y_span)
    
    letter_df = pd.DataFrame({
        group_var_name: list(cld_letters.keys()),
        'letter': list(cld_letters.values()),
        'y_pos': [letter_y[g] for g in cld_letters.keys()]
    })
    
    # Create chart
    base = alt.Chart(plot_df).encode(
        x=alt.X(f'{group_var_name}:N', title=group_var_name.replace('_', ' ')),
        color=alt.Color(f'{group_var_name}:N', legend=None)
    )
    
    boxplot = base.mark_boxplot(size=30, extent='min-max').encode(
        y=alt.Y(f'{dependent_var_name}:Q', 
                title=dependent_var_name.replace('_', ' ').replace('perc', '%'))
    )
    
    points = base.mark_circle(size=30, opacity=0.7).encode(
        x=alt.X('jitter:Q', axis=None, scale=alt.Scale(domain=[-1, 1])),
        y=f'{dependent_var_name}:Q'
    )
    
    letters = alt.Chart(letter_df).mark_text(
        dy=-15, 
        fontSize=14,
        fontWeight='bold'
    ).encode(
        x=f'{group_var_name}:N',
        y=alt.Y('y_pos:Q', axis=None),
        text='letter:N'
    )
    
    chart = (boxplot + points + letters).properties(
        width=600,
        height=400
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    st.altair_chart(chart, use_container_width=True)

# --- Streamlit App with Enhanced Features ---
st.title("Vermicompost Statistical Analysis")
st.markdown("""
    This tool performs statistical analysis comparing different material groups.
    Upload your data or use sample data to get started.
""")

# Data loading with upload option
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded data: {uploaded_file.name}")
    else:
        st.info("Using sample data")
        df = pd.DataFrame({
            'Material_Group': np.repeat(['Group A', 'Group B', 'Group C'], 30),
            'N_perc': np.concatenate([
                np.random.normal(3.0, 0.3, 30),  # Higher mean for Group A
                np.random.normal(2.0, 0.4, 30),
                np.random.normal(2.5, 0.3, 30)
            ]),
            'P_perc': np.random.normal(0.5, 0.1, 90),
            'K_perc': np.random.normal(1.2, 0.2, 90),
            'C_N_Ratio_final': np.concatenate([
                np.random.normal(12, 2, 30),
                np.random.normal(18, 3, 30),
                np.random.normal(15, 2, 30)
            ]),
            'pH': np.random.normal(7.0, 0.5, 90)
        })
    
    # Variable selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    group_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    if not group_cols:
        group_cols = df.columns.tolist()
    
    selected_var = st.selectbox("Select measurement variable", numeric_cols)
    group_var = st.selectbox("Select group variable", group_cols)

# Run analysis
if st.sidebar.button("Run Analysis"):
    run_statistical_analysis_and_plot(df, selected_var, group_var)
else:
    st.info("Select variables and click 'Run Analysis' in the sidebar")
    st.dataframe(df.head().style.format("{:.2f}"))
