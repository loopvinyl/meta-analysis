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
# This function assigns letters to groups based on their significance differences.
# Groups sharing a letter are not significantly different.
def get_compact_letter_display(p_values_matrix, group_names):
    # Based on a common algorithm for compact letter display
    # (e.g., similar to multcompView in R)
    
    # Sort groups to ensure consistent assignment
    sorted_groups = sorted(group_names)
    
    # Create a list of tuples (group1, group2) for significant differences (p < 0.05)
    significant_pairs = []
    
    # Ensure p_values_matrix is a DataFrame for easier slicing
    if not isinstance(p_values_matrix, pd.DataFrame):
        st.error("Internal Error: P-value matrix for CLD is not in the expected format (DataFrame).")
        return {group: '' for group in group_names} # Return empty if format is wrong
    
    for i in range(len(sorted_groups)):
        for j in range(i + 1, len(sorted_groups)):
            g1 = sorted_groups[i]
            g2 = sorted_groups[j]
            
            # Extract p-value from the symmetric matrix (either [g1,g2] or [g2,g1])
            # Handle potential KeyError if group not found (shouldn't happen with correct group_names)
            try:
                p_val = p_values_matrix.loc[g1, g2]
            except KeyError:
                try:
                    p_val = p_values_matrix.loc[g2, g1]
                except KeyError:
                    p_val = 1.0 # Assume no significant difference if not found
            
            if p_val < 0.05: # Assuming alpha = 0.05 for significance
                significant_pairs.append(tuple(sorted(tuple([g1, g2])))) # Store as sorted tuple

    # This is a simplified greedy algorithm for CLD.
    # More robust algorithms exist but are complex to implement from scratch.
    # This aims to produce valid letters but might not be the 'most compact' in all scenarios.
    
    # Initialize letters for each group as empty lists
    group_letters = {g: [] for g in group_names}
    
    # A list to keep track of clusters of non-significantly different groups
    clusters = []

    # Iterate through sorted groups to form clusters
    for g in sorted_groups:
        assigned_to_existing_cluster = False
        for cluster in clusters:
            can_add_to_cluster = True
            for member in cluster:
                # Check if 'g' is significantly different from any member in the current cluster
                if tuple(sorted((g, member))) in significant_pairs:
                    can_add_to_cluster = False
                    break
            if can_add_to_cluster:
                cluster.add(g) # Add 'g' to this cluster
                assigned_to_existing_cluster = True
                break
        if not assigned_to_existing_cluster:
            clusters.append({g}) # Create a new cluster for 'g'
            
    # Assign letters based on the formed clusters
    # Use ascii_lowercase for letters 'a', 'b', 'c', ...
    available_letters = list(string.ascii_lowercase)
    
    for i, cluster in enumerate(clusters):
        if i < len(available_letters):
            letter = available_letters[i]
            for group in cluster:
                group_letters[group].append(letter)
        else:
            # Handle case where too many clusters for single letters, fallback or extend logic
            st.warning("Too many distinct clusters for single letter assignment. CLD might be incomplete.")
            for group in cluster:
                group_letters[group].append(str(i)) # Assign numeric indicator

    # Combine all assigned letters for each group, sort them, and join into a string
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
            # Using scipy.stats.levene directly on groups
            groups_for_levene = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() for g in data_clean[group_var_name].unique()]
            # Filter out empty groups if any
            groups_for_levene = [g for g in groups_for_levene if len(g) > 0] 
            
            if len(groups_for_levene) < 2:
                st.info("Not enough groups with data for Levene's Test.")
                homogeneous_variances = False
            else:
                stat, p_levene = stats.levene(*groups_for_levene)
                st.write(f"Levene's Statistic: {stat:.3f}, p-value: {p_levene:.3f}")
                if p_levene < 0.05:
                    st.warning("Variances are **NOT homogeneous** (p < 0.05). This may suggest using non-parametric tests or corrections.")
                    homogeneous_variances = False
                else:
                    st.success("Variances **ARE homogeneous** (p >= 0.05).")
                    homogeneous_variances = True
        except Exception as e:
            st.error(f"Could not perform Levene's Test: {e}")
            homogeneous_variances = False # Assume non-homogeneous if test fails


    # 2. Normality Test (Shapiro-Wilk by group)
    with st.expander(f"Normality Test (Shapiro-Wilk by group) for {dependent_var_name}"):
        st.write("Evaluates if data in each group follows a normal distribution.")
        shapiro_results = []
        normality_by_group = True
        for group in data_clean[group_var_name].unique():
            group_data = data_clean[data_clean[group_var_name] == group][dependent_var_name].dropna()
            if len(group_data) >= 3: # Shapiro-Wilk requires at least 3 data points
                stat_shapiro, p_shapiro = stats.shapiro(group_data)
                shapiro_results.append({'Group': group, 'N': len(group_data), 'Statistic': stat_shapiro, 'p-value': p_shapiro})
                if p_shapiro < 0.05:
                    normality_by_group = False
            else:
                shapiro_results.append({'Group': group, 'N': len(group_data), 'Statistic': np.nan, 'p-value': np.nan})
                st.info(f"Group '{group}' has less than 3 data points for Shapiro-Wilk Test. Not tested for normality.")

        shapiro_df = pd.DataFrame(shapiro_results)
        if not shapiro_df.empty:
            st.dataframe(shapiro_df.set_index('Group'))
        else:
            st.info("No groups with enough data for normality test.")


        if not shapiro_df['p-value'].isnull().all(): # Check if any p-values were calculated
            if not normality_by_group:
                st.warning("At least one group **DOES NOT follow a normal distribution** (p < 0.05).")
            else:
                st.success("All tested groups **follow a normal distribution** (p >= 0.05).")
        else:
            st.info("Normality could not be tested for any group (N too small).")

    # 3. Select and Execute Statistical Tests
    st.markdown("#### Statistical Test Results")
    post_hoc_results_df = None
    cld_letters = {} # Initialize cld_letters
    
    num_groups = data_clean[group_var_name].nunique()

    if num_groups < 2:
        st.info("Only one or no group found for comparison. Statistical tests not applicable.")
        return # Exit the function if not enough groups

    if homogeneous_variances and normality_by_group:
        st.info("Conditions met: Using **parametric ANOVA**.")
        with st.expander("ANOVA Results"):
            try:
                formula = f'{dependent_var_name} ~ C({group_var_name})'
                model = ols(formula, data=data_clean).fit()
                anova_table = anova_lm(model, typ=2) # Type 2 ANOVA sum of squares
                st.write("ANOVA Table:")
                st.dataframe(anova_table)

                if anova_table['PR(>F)'].iloc[0] < 0.05: # P-value for the group effect
                    st.success(f"ANOVA is **SIGNIFICANT** (p < 0.05), indicating a difference between groups.")
                    
                    st.markdown("##### Post-hoc Test: Tukey HSD")
                    # Tukey HSD Post-hoc
                    tukey_result = pairwise_tukeyhsd(endog=data_clean[dependent_var_name],
                                                     groups=data_clean[group_var_name],
                                                     alpha=0.05)
                    st.write(tukey_result)

                    # Prepare p-values for CLD from Tukey result
                    # Create a symmetric p-value matrix for CLD function
                    groups_unique = data_clean[group_var_name].unique()
                    p_matrix = pd.DataFrame(np.ones((len(groups_unique), len(groups_unique))), 
                                            index=groups_unique, 
                                            columns=groups_unique)
                    for idx, row in pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0]).iterrows():
                        group1 = row['group1']
                        group2 = row['group2']
                        p_adj = row['p-adj']
                        p_matrix.loc[group1, group2] = p_adj
                        p_matrix.loc[group2, group1] = p_adj # Symmetric
                    
                    cld_letters = get_compact_letter_display(p_matrix, groups_unique)
                    st.write("#### Significance Letters (CLD):")
                    st.write(cld_letters)
                else:
                    st.info(f"ANOVA is **NOT significant** (p >= 0.05). No statistical difference detected between groups.")

            except Exception as e:
                st.error(f"Error performing ANOVA: {e}")

    else:
        st.info("Conditions VIOLATED: Using **Kruskal-Wallis Test** (non-parametric).")
        with st.expander("Kruskal-Wallis Results"):
            try:
                # Kruskal-Wallis Test
                groups_kruskal = [data_clean[dependent_var_name][data_clean[group_var_name] == g].dropna() for g in data_clean[group_var_name].unique()]
                groups_kruskal = [g for g in groups_kruskal if len(g) > 0] # Filter out empty groups
                
                if len(groups_kruskal) < 2:
                    st.info("Not enough groups with data for Kruskal-Wallis Test.")
                else:
                    stat_kruskal, p_kruskal = stats.kruskal(*groups_kruskal)
                    st.write(f"Kruskal-Wallis H Statistic: {stat_kruskal:.3f}, p-value: {p_kruskal:.3f}")

                    if p_kruskal < 0.05:
                        st.success(f"Kruskal-Wallis test is **SIGNIFICANT** (p < 0.05), indicating a difference between groups.")
                        
                        st.markdown("##### Post-hoc Test: Dunn with Bonferroni correction")
                        # Dunn's Post-hoc Test (scikit-posthocs returns a DataFrame directly)
                        dunn_result = sp.posthoc_dunn(data_clean, val_col=dependent_var_name,
                                                      group_col=group_var_name, p_adjust='bonferroni')
                        st.dataframe(dunn_result)

                        # Prepare p-values for CLD (Dunn's results are already a matrix)
                        cld_letters = get_compact_letter_display(dunn_result, data_clean[group_var_name].unique())
                        st.write("#### Significance Letters (CLD):")
                        st.write(cld_letters)
                        
                    else:
                        st.info(f"Kruskal-Wallis test is **NOT significant** (p >= 0.05). No statistical difference detected between groups.")
            except Exception as e:
                st.error(f"Error performing Kruskal-Wallis: {e}")

    # --- Visualization (Boxplot with Jitter and CLD using Matplotlib/Seaborn) ---
    st.markdown("#### Data Visualization (Boxplot with Jitter)")
    
    # Prepare data for plotting
    plot_df = data_clean.copy()

    # Calculate overall min/max for Y-axis for consistent scaling
    min_y_overall = plot_df[dependent_var_name].min()
    max_y_overall = plot_df[dependent_var_name].max()
    
    # Calculate y-positions for CLD letters
    plot_df_max_y = pd.DataFrame() # Initialize as empty
    if cld_letters: # Only proceed if CLD was calculated and is not empty
        # Get max value for each group
        max_vals_per_group = plot_df.groupby(group_var_name)[dependent_var_name].max().reset_index()
        
        # Merge with CLD letters using the actual group_var_name
        plot_df_max_y = max_vals_per_group.merge(pd.DataFrame(cld_letters.items(), columns=[group_var_name, 'cld_letter']), on=group_var_name)
        
        # Calculate y_pos with a buffer above the max value of each group
        buffer = (max_y_overall - min_y_overall) * 0.08 # 8% buffer
        plot_df_max_y['y_pos'] = plot_df_max_y[dependent_var_name] + buffer

        # Ensure y_pos and cld_letter are not NaN for plotting
        plot_df_max_y = plot_df_max_y.dropna(subset=['y_pos', 'cld_letter']).copy()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot
    sns.boxplot(data=plot_df, x=group_var_name, y=dependent_var_name, ax=ax)

    # Jitter (stripplot)
    sns.stripplot(data=plot_df, x=group_var_name, y=dependent_var_name,
                  color='black', size=5, jitter=0.2, ax=ax, linewidth=0.5)

    # Add CLD letters if available and plot_df_max_y is not empty
    if not plot_df_max_y.empty:
        for _, row in plot_df_max_y.iterrows():
            ax.text(x=row[group_var_name], y=row['y_pos'], s=row['cld_letter'],
                    ha='center', va='bottom', fontsize=12, color='red', weight='bold') # va='bottom' for better placement

    ax.set_title(f"Distribution of {dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')} by Material Group", fontsize=16)
    ax.set_xlabel("Material Group", fontsize=12)
    ax.set_ylabel(dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', ''), fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
    
    # Adjust y-axis limits to accommodate CLD letters
    if not plot_df_max_y.empty:
        max_y_for_limit = plot_df_max_y['y_pos'].max()
    else:
        max_y_for_limit = max_y_overall
    
    # Add a bit more buffer to the top limit for visual spacing
    plt.ylim(min_y_overall, max_y_for_limit * 1.05) 

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    st.pyplot(fig) # Display the plot in Streamlit

    # --- Interpretation of Results ---
    with st.expander(f"✨ Detailed Interpretation for {dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}"):
        st.markdown("The interpretation of the results should consider the tests for homogeneity of variance and normality, and then the results of ANOVA or Kruskal-Wallis and their respective post-hoc tests.")
        
        st.markdown("##### Homogeneity of Variances (Levene's Test):")
        if homogeneous_variances:
            st.markdown("- **P > 0.05**: Variances between groups are considered **equal** (homogeneous). This is good for ANOVA.")
        else:
            st.markdown("- **P < 0.05**: Variances between groups are considered **different** (non-homogeneous). This suggests that ANOVA may not be the most appropriate test, and Kruskal-Wallis (non-parametric) is a robust alternative.")
        
        st.markdown("##### Normality (Shapiro-Wilk per Group):")
        if normality_by_group:
            st.markdown("- **P > 0.05 for all groups**: Data in each group follows a **normal** distribution. This is an assumption for ANOVA.")
        else:
            st.markdown("- **P < 0.05 for one or more groups**: Data in at least one group **does not follow a normal distribution**. This also points to using non-parametric tests like Kruskal-Wallis.")
            
        st.markdown("##### Main Test (ANOVA or Kruskal-Wallis):")
        if homogeneous_variances and normality_by_group:
            st.markdown("- **If ANOVA P < 0.05**: There is a statistically significant difference between the means of the groups for the analyzed variable. Proceed to Tukey HSD to identify which groups are different.")
            st.markdown("- **If ANOVA P >= 0.05**: No evidence of statistical difference between group means.")
        else:
            st.markdown("- **If Kruskal-Wallis P < 0.05**: There is a statistically significant difference between the medians (or distributions) of the groups for the analyzed variable. Proceed to Dunn's Test to identify which groups are different.")
            st.markdown("- **If Kruskal-Wallis P >= 0.05**: No evidence of statistical difference between group medians/distributions.")

        st.markdown("##### Post-hoc Test (Tukey HSD or Dunn):")
        if cld_letters:
            st.markdown("The **significance letters (CLD)** in the chart indicate group clustering in a compact way:")
            st.markdown("- Groups that **share the same letter** are not statistically different from each other.")
            st.markdown("- Groups that **DO NOT share any letter** are statistically different from each other.")
            st.markdown("For example, if groups have letters 'a', 'ab', 'b':")
            st.markdown("  - 'a' is different from 'b'.")
            st.markdown("  - 'ab' is not different from 'a' and not different from 'b'.")
        else:
            st.markdown("No post-hoc test was performed, as the main test (ANOVA or Kruskal-Wallis) was not significant, or there were not enough groups for comparison.")

# --- 2. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Vermicompost Meta-analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Updated App Title ---
st.title("🔬 Meta-analysis of Vermicompost: Waste Type Impacts on N, K, and pH")
st.markdown("This interactive application allows you to explore the statistical test results for Nitrogen, Phosphorus, Potassium, pH, and C/N Ratio in vermicomposts, grouped by source material.")
st.markdown("---")

# --- 1. Data Loading ---
# The dados_vermicomposto.xlsx file should be in the same folder as this script.
try:
    df = pd.read_excel('dados_vermicomposto.xlsx')
except FileNotFoundError:
    st.error("Error: 'dados_vermicomposto.xlsx' file not found. Please ensure it is in the same folder as this script.")
    st.stop()

# --- Column Renaming Map ---
# Map columns from your Excel file to the standardized names used in the script.
# This allows the app to work without you changing the Excel column names.
column_rename_map = {
    'Material de Origem do Vermicomposto': 'Source_Material',
    'N (%)': 'N_perc',
    'P (%)': 'P_perc',
    'K (%)': 'K_perc',
    'pH final': 'pH_final',
    'CN_Ratio_final': 'C_N_Ratio_final'
}

# --- Apply Renaming and Check for Missing Columns ---
# We check if the *original* column exists before attempting to rename.
# This makes it more robust against partially missing columns in the Excel.
cols_to_rename_existing = {
    old_name: new_name for old_name, new_name in column_rename_map.items() 
    if old_name in df.columns
}
df.rename(columns=cols_to_rename_existing, inplace=True)

# Now, check for the *expected standardized names*
required_standard_cols = list(column_rename_map.values())
missing_standard_cols = [col for col in required_standard_cols if col not in df.columns]

if missing_standard_cols:
    st.error(f"Error: After renaming, the following required columns were not found or could not be mapped in the data file: {', '.join(missing_standard_cols)}. Please check the original column names in your Excel: {', '.join([k for k, v in column_rename_map.items() if v in missing_standard_cols])}.")
    st.stop()


# Categorization Material_Group (as in R code)
df['Material_Group'] = "Uncategorized" # Default value

# Categorize based on keywords. Case-insensitive.
df.loc[df['Source_Material'].str.contains("Pineapple|Abacaxi|Banana Leaf|Food Waste|Kitchen Waste", case=False, na=False), 'Material_Group'] = "Fruit & Vegetable Waste"
df.loc[df['Source_Material'].str.contains("Coffee|SCG", case=False, na=False), 'Material_Group'] = "Coffee Waste" 
df.loc[df['Source_Material'].str.contains("Manure|Dung|Cattle Manure|Cow Manure|Pig Manure|Bagasse:|B0|B25|B50|B75", case=False, na=False), 'Material_Group'] = "Manure & Related"
df.loc[df['Source_Material'].str.contains("Grass Clippings|Water Hyacinth|Parthenium|Bagasse \\(100:0\\)|Pure Vermicompost|Matka khad|Kitchen-Yard Waste", case=False, na=False), 'Material_Group'] = "Diverse Plant Waste"
df.loc[df['Source_Material'].str.contains("Newspaper|Paper Waste|Cardboard", case=False, na=False), 'Material_Group'] = "Paper & Cellulose Waste"

df['Material_Group'] = df['Material_Group'].astype('category')

# --- NOVO BLOCO: Visualize Tipos de Materiais por Grupo ---
st.markdown("---")
st.subheader("🔍 Explore Materiais de Origem em Cada Grupo")
with st.expander("Clique para ver quais 'Materiais de Origem' pertencem a cada 'Grupo de Material'"):
    # Crie uma lista única dos grupos e seus materiais de origem correspondentes
    # Usamos df.dropna() para garantir que apenas dados válidos sejam considerados
    grouped_materials = df.groupby('Material_Group')['Source_Material'].unique()
    
    # Exiba cada grupo e seus materiais
    for group, materials in grouped_materials.items():
        st.write(f"**{group}**:")
        if len(materials) > 0:
            st.write(", ".join(materials))
        else:
            st.write("Nenhum material encontrado para este grupo.")
st.markdown("---")
# --- FIM DO NOVO BLOCO ---

# Variables for analysis (these use the standardized internal names)
numerical_variables = {
    "Nitrogen (%)": "N_perc",
    "Phosphorus (%)": "P_perc",
    "Potassium (%)": "K_perc",
    "Final pH": "pH_final",
    "Final C/N Ratio": "C_N_Ratio_final"
}

st.sidebar.header("Select Variable for Analysis")
selected_variable_display_name = st.sidebar.selectbox(
    "Choose the numerical variable to analyze:",
    list(numerical_variables.keys())
)

var_name = numerical_variables[selected_variable_display_name]
group_col = "Material_Group" # The code now creates and uses this column

# --- Execute Analysis and Plot ---
run_statistical_analysis_and_plot(df, var_name, group_col)
