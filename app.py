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

# --- Helper Function for Compact Letter Display (CLD) ---
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
            try:
                p_val = p_values_matrix.loc[g1, g2]
            except KeyError:
                try:
                    p_val = p_values_matrix.loc[g2, g1]
                except KeyError:
                    p_val = 1.0
            
            if p_val < 0.05:
                significant_pairs.append(tuple(sorted(tuple([g1, g2]))))

    group_letters = {g: [] for g in group_names}
    clusters = []

    for g in sorted_groups:
        assigned_to_existing_cluster = False
        for cluster in clusters:
            can_add_to_cluster = True
            for member in cluster:
                if tuple(sorted((g, member))) in significant_pairs:
                    can_add_to_cluster = False
                    break
            if can_add_to_cluster:
                cluster.add(g)
                assigned_to_existing_cluster = True
                break
        if not assigned_to_existing_cluster:
            clusters.append({g})
            
    available_letters = list(string.ascii_lowercase)
    
    for i, cluster in enumerate(clusters):
        if i < len(available_letters):
            letter = available_letters[i]
            for group in cluster:
                group_letters[group].append(letter)
        else:
            st.warning("Too many distinct clusters for single letter assignment. CLD might be incomplete.")
            for group in cluster:
                group_letters[group].append(str(i))

    final_letters = {group: "".join(sorted(list(set(letters)))) for group, letters in group_letters.items()}
    return final_letters


# --- Function to run statistical analysis and display results ---
def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    st.markdown(f"#### Analysis for: **{dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}**")

    # Clean data (remove NA for this specific analysis)
    data_clean = data.dropna(subset=[dependent_var_name, group_var_name]).copy()

    # Check if there's enough data for analysis
    if data_clean[group_var_name].nunique() < 2:
        st.warning(f"Not enough groups ({data_clean[group_var_name].nunique()}) for statistical analysis of {dependent_var_name}.")
        return

    # 1. Homogeneity of Variance Test (Levene's Test)
    with st.expander(f"Homogeneity of Variance Test (Levene's Test) for {dependent_var_name}"):
        st.write("Evaluates if group variances are equal.")
        try:
            groups_for_levene = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() 
                                for g in data_clean[group_var_name].unique()]
            groups_for_levene = [g for g in groups_for_levene if len(g) > 0] 
            
            if len(groups_for_levene) < 2:
                st.info("Not enough groups with data for Levene's Test.")
                homogeneous_variances = False
            else:
                stat, p_levene = stats.levene(*groups_for_levene)
                st.write(f"Levene's Statistic: {stat:.3f}, p-value: {p_levene:.3f}")
                homogeneous_variances = p_levene >= 0.05
                if p_levene < 0.05:
                    st.warning("Variances are NOT homogeneous (p < 0.05). Consider non-parametric tests.")
                else:
                    st.success("Variances ARE homogeneous (p >= 0.05).")
        except Exception as e:
            st.error(f"Could not perform Levene's Test: {e}")
            homogeneous_variances = False

    # 2. Normality Test (Shapiro-Wilk by group)
    with st.expander(f"Normality Test (Shapiro-Wilk by group) for {dependent_var_name}"):
        st.write("Evaluates if data in each group follows a normal distribution.")
        shapiro_results = []
        normality_by_group = True
        for group in data_clean[group_var_name].unique():
            group_data = data_clean[data_clean[group_var_name] == group][dependent_var_name].dropna()
            if len(group_data) >= 3:
                stat_shapiro, p_shapiro = stats.shapiro(group_data)
                shapiro_results.append({'Group': group, 'N': len(group_data), 
                                       'Statistic': stat_shapiro, 'p-value': p_shapiro})
                if p_shapiro < 0.05:
                    normality_by_group = False
            else:
                shapiro_results.append({'Group': group, 'N': len(group_data), 
                                       'Statistic': np.nan, 'p-value': np.nan})
                st.info(f"Group '{group}' has less than 3 data points. Normality not tested.")

        shapiro_df = pd.DataFrame(shapiro_results)
        if not shapiro_df.empty:
            st.dataframe(shapiro_df.set_index('Group'))
        
        if not shapiro_df['p-value'].isnull().all():
            if not normality_by_group:
                st.warning("At least one group DOES NOT follow a normal distribution (p < 0.05).")
            else:
                st.success("All tested groups follow a normal distribution (p >= 0.05).")
        else:
            st.info("Normality could not be tested for any group (N too small).")

    # 3. Select and Execute Statistical Tests
    st.markdown("#### Statistical Test Results")
    cld_letters = {}
    num_groups = data_clean[group_var_name].nunique()

    if num_groups < 2:
        st.info("Only one group found. Statistical tests not applicable.")
        return

    if homogeneous_variances and normality_by_group:
        st.info("Conditions met: Using parametric ANOVA.")
        with st.expander("ANOVA Results"):
            try:
                formula = f'{dependent_var_name} ~ C({group_var_name})'
                model = ols(formula, data=data_clean).fit()
                anova_table = anova_lm(model, typ=2)
                st.write("ANOVA Table:")
                st.dataframe(anova_table)

                if anova_table['PR(>F)'].iloc[0] < 0.05:
                    st.success("ANOVA is SIGNIFICANT (p < 0.05). Differences between groups detected.")
                    
                    st.markdown("##### Post-hoc Test: Tukey HSD")
                    tukey_result = pairwise_tukeyhsd(endog=data_clean[dependent_var_name],
                                                     groups=data_clean[group_var_name],
                                                     alpha=0.05)
                    st.write(tukey_result)

                    # Prepare p-values for CLD
                    groups_unique = data_clean[group_var_name].unique()
                    p_matrix = pd.DataFrame(np.ones((len(groups_unique), len(groups_unique))), 
                                            index=groups_unique, 
                                            columns=groups_unique)
                    tukey_df = pd.DataFrame(tukey_result._results_table.data[1:], 
                                           columns=tukey_result._results_table.data[0])
                    for _, row in tukey_df.iterrows():
                        group1, group2, p_adj = row['group1'], row['group2'], row['p-adj']
                        p_matrix.loc[group1, group2] = p_adj
                        p_matrix.loc[group2, group1] = p_adj
                    
                    cld_letters = get_compact_letter_display(p_matrix, groups_unique)
                    st.write("#### Significance Letters (CLD):")
                    st.write(cld_letters)
                else:
                    st.info("ANOVA is NOT significant (p >= 0.05). No statistical difference detected.")
            except Exception as e:
                st.error(f"Error performing ANOVA: {e}")

    else:
        st.info("Conditions violated: Using non-parametric Kruskal-Wallis Test.")
        with st.expander("Kruskal-Wallis Results"):
            try:
                groups_kruskal = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() 
                                 for g in data_clean[group_var_name].unique()]
                groups_kruskal = [g for g in groups_kruskal if len(g) > 0]
                
                if len(groups_kruskal) < 2:
                    st.info("Not enough groups with data for Kruskal-Wallis Test.")
                else:
                    stat_kruskal, p_kruskal = stats.kruskal(*groups_kruskal)
                    st.write(f"Kruskal-Wallis H Statistic: {stat_kruskal:.3f}, p-value: {p_kruskal:.3f}")

                    if p_kruskal < 0.05:
                        st.success("Kruskal-Wallis test is SIGNIFICANT (p < 0.05). Differences between groups detected.")
                        
                        st.markdown("##### Post-hoc Test: Dunn with Bonferroni correction")
                        dunn_result = sp.posthoc_dunn(data_clean, val_col=dependent_var_name,
                                                      group_col=group_var_name, p_adjust='bonferroni')
                        st.dataframe(dunn_result)

                        cld_letters = get_compact_letter_display(dunn_result, data_clean[group_var_name].unique())
                        st.write("#### Significance Letters (CLD):")
                        st.write(cld_letters)
                    else:
                        st.info("Kruskal-Wallis test is NOT significant (p >= 0.05). No statistical difference detected.")
            except Exception as e:
                st.error(f"Error performing Kruskal-Wallis: {e}")

    # --- Visualization (Boxplot with Jitter and CLD) ---
    st.markdown("#### Data Visualization (Boxplot with Jitter)")
    plot_df = data_clean.copy()
    min_y_overall = plot_df[dependent_var_name].min()
    max_y_overall = plot_df[dependent_var_name].max()
    
    plot_df_max_y = pd.DataFrame()
    if cld_letters:
        max_vals_per_group = plot_df.groupby(group_var_name)[dependent_var_name].max().reset_index()
        plot_df_max_y = max_vals_per_group.merge(
            pd.DataFrame(cld_letters.items(), columns=[group_var_name, 'cld_letter']), 
            on=group_var_name
        )
        buffer = (max_y_overall - min_y_overall) * 0.08
        plot_df_max_y['y_pos'] = plot_df_max_y[dependent_var_name] + buffer
        plot_df_max_y = plot_df_max_y.dropna(subset=['y_pos', 'cld_letter']).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x=group_var_name, y=dependent_var_name, ax=ax)
    sns.stripplot(data=plot_df, x=group_var_name, y=dependent_var_name,
                  color='black', size=5, jitter=0.2, ax=ax, linewidth=0.5)

    if not plot_df_max_y.empty:
        for _, row in plot_df_max_y.iterrows():
            ax.text(x=row[group_var_name], y=row['y_pos'], s=row['cld_letter'],
                    ha='center', va='bottom', fontsize=12, color='red', weight='bold')

    ax.set_title(f"Distribution of {dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')} by Material Group", fontsize=16)
    ax.set_xlabel("Material Group", fontsize=12)
    ax.set_ylabel(dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', ''), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    max_y_for_limit = plot_df_max_y['y_pos'].max() if not plot_df_max_y.empty else max_y_overall
    plt.ylim(min_y_overall, max_y_for_limit * 1.05) 

    plt.tight_layout()
    st.pyplot(fig)

    # --- Interpretation of Results ---
    with st.expander(f"✨ Detailed Interpretation"):
        st.markdown("The interpretation considers tests for homogeneity of variance, normality, and main test results.")
        
        st.markdown("##### Homogeneity of Variances (Levene's Test):")
        st.markdown(f"- {'Variances ARE homogeneous' if homogeneous_variances else 'Variances are NOT homogeneous'}")
        
        st.markdown("##### Normality (Shapiro-Wilk per Group):")
        st.markdown(f"- {'All groups follow normal distribution' if normality_by_group else 'At least one group does not follow normal distribution'}")
            
        st.markdown("##### Main Test Results:")
        if homogeneous_variances and normality_by_group:
            st.markdown("- Parametric ANOVA was used")
        else:
            st.markdown("- Non-parametric Kruskal-Wallis was used")
            
        if cld_letters:
            st.markdown("##### Significance Letters (CLD):")
            st.markdown("- Groups sharing the same letter are not statistically different")
            st.markdown("- Groups with different letters are statistically different")
        else:
            st.markdown("No significant differences detected between groups")

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Vermicompost Meta-analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Meta-analysis of Vermicompost: Waste Type Impacts on Nutrients")
st.markdown("Explore statistical results for Nitrogen, Phosphorus, Potassium, pH, and C/N Ratio in vermicomposts")
st.markdown("---")

# --- 1. Data Loading with New File Name ---
try:
    df = pd.read_excel('excelv2.xlsx')  # Updated file name
except FileNotFoundError:
    st.error("Error: 'excelv2.xlsx' file not found. Please ensure it's in the same folder.")
    st.stop()

# --- Updated Column Renaming for New Table Structure ---
column_rename_map = {
    'N (%)': 'N_perc',
    'P (%)': 'P_perc',
    'K (%)': 'K_perc',
    'pH_final': 'pH_final',
    'CN_Ratio_final': 'C_N_Ratio_final'
}

# Apply renaming only to existing columns
cols_to_rename = {}
for old_name, new_name in column_rename_map.items():
    if old_name in df.columns:
        cols_to_rename[old_name] = new_name
df.rename(columns=cols_to_rename, inplace=True)

# Check for required columns
required_cols = list(column_rename_map.values()) + ['Source_Material']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns: {', '.join(missing_cols)}. Please check your Excel file structure.")
    st.stop()

# --- Enhanced Material Group Categorization ---
df['Material_Group'] = "Uncategorized"

# Define categorization rules with improved keywords
category_rules = {
    "Pineapple Waste": "Pineapple|Abacaxi|Pineapple peels",
    "Coffee Waste": "Coffee|SCG|Borra de Café|Spent Coffee Grounds",
    "Vegetable Waste": "Vegetable|Grass|Water Hyacinth|Parthenium|Cotton|Bagasse|Crop residue|Weeds",
    "Manure & Related": "Manure|Dung|Cattle|Cow|Pig|Horse|Bovine|Matka khad|Pure Vermicompost|CD|PM|B0|B25|B50|B75|VR",
    "Newspaper Waste": "Newspaper|Paper|Cardboard",
    "Food Waste": "Food|Kitchen",
    "Banana": "Banana Leaf"
}

for category, keywords in category_rules.items():
    df.loc[df['Source_Material'].str.contains(keywords, case=False, na=False), 'Material_Group'] = category

# Filter out non-vermicompost entries
if 'Additional Observations' in df.columns:
    df = df[~df['Additional Observations'].str.contains('Not vermicompost|Drum compost', case=False, na=False)]

# --- Enhanced Material Exploration ---
st.markdown("---")
st.subheader("🔍 Source Materials by Group and Article")
with st.expander("View detailed material categorization"):
    if 'Article (Authors, Year)' in df.columns:
        grouped = df.groupby('Material_Group')
        for name, group in grouped:
            unique_combos = group[['Source_Material', 'Article (Authors, Year)']].drop_duplicates()
            if not unique_combos.empty:
                st.write(f"**{name}**")
                for _, row in unique_combos.iterrows():
                    st.write(f"- {row['Source_Material']} ({row['Article (Authors, Year)'})")
    else:
        st.warning("Article information not available in dataset")
st.markdown("---")

# --- Analysis Variables ---
numerical_variables = {
    "Nitrogen (%)": "N_perc",
    "Phosphorus (%)": "P_perc",
    "Potassium (%)": "K_perc",
    "Final pH": "pH_final",
    "Final C/N Ratio": "C_N_Ratio_final"
}

# --- Sidebar Controls ---
st.sidebar.header("Analysis Parameters")
selected_variable = st.sidebar.selectbox(
    "Select variable to analyze:",
    list(numerical_variables.keys())
)

# --- Execute Analysis ---
run_statistical_analysis_and_plot(df, numerical_variables[selected_variable], "Material_Group")
