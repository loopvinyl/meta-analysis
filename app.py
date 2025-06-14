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

# --- Material Group Categorization Function (ATUALIZADA) ---
def assign_material_group(source):
    if pd.isna(source):
        return "Uncategorized"
    source = str(source).lower()
    
    # 1. Manure-Based (Todos os tipos de esterco + misturas)
    manure_keywords = ["manure", "dung", "cattle", "cow", "bovine", "cd", "vr", "fezes", "estrume", "gado", "vaca"]
    if any(kw in source for kw in manure_keywords):
        return "Manure-Based"
    
    # 2. Coffee Waste (Prioridade alta - mantido separado)
    coffee_keywords = ["coffee", "scg", "borra", "café"]
    if any(kw in source for kw in coffee_keywords):
        return "Coffee Waste"
    
    # 3. Agro-Industrial Waste (Resíduos de frutas + alimentos + cultivos)
    agro_industrial_keywords = ["pineapple", "abacaxi", "fruit", "fruta", "peels", 
                                "food", "kitchen", "alimento", 
                                "bagasse", "crop", "residue", "resíduo", "straw", "palha", "sugarcane", "bagaço"]
    if any(kw in source for kw in agro_industrial_keywords):
        return "Agro-Industrial Waste"
    
    # 4. Plant Waste (Materiais vegetais frescos)
    plant_keywords = ["vegetable", "grass", "water hyacinth", "weeds", "parthenium", "green", "verde", "hortaliças"]
    if any(kw in source for kw in plant_keywords):
        return "Plant Waste"
    
    # 5. Cellulosic Waste
    cellulosic_keywords = ["cardboard", "paper", "filters", "filtro", "cellulose", "papel", "papelão"]
    if any(kw in source for kw in cellulosic_keywords):
        return "Cellulosic Waste"
    
    return "Uncategorized"

# --- Get category description (ATUALIZADA) ---
def get_category_description(category):
    descriptions = {
        "Manure-Based": "Todos os vermicompostos com base em esterco animal (puros ou misturas)",
        "Coffee Waste": "Resíduos de café processado (borra) sem esterco",
        "Agro-Industrial Waste": "Resíduos de processamento agrícola e industrial (frutas, alimentos, cultivos)",
        "Plant Waste": "Materiais vegetais frescos (hortaliças, grama, plantas aquáticas)",
        "Cellulosic Waste": "Materiais ricos em celulose (papelão, filtros, papel)"
    }
    return descriptions.get(category, "Sem descrição disponível")

# --- Function to run statistical analysis and display results ---
def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    st.markdown(f"#### Analysis for: **{dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')}**")

    # Convert to numeric and clean
    data[dependent_var_name] = pd.to_numeric(data[dependent_var_name], errors='coerce')
    data_clean = data.dropna(subset=[dependent_var_name, group_var_name]).copy()
    
    if data_clean.empty:
        st.warning("No valid data available for analysis after cleaning.")
        return
        
    # Check if there's enough data for analysis
    if data_clean[group_var_name].nunique() < 2:
        st.warning(f"Not enough groups ({data_clean[group_var_name].nunique()}) for statistical analysis.")
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
    
    if plot_df.empty:
        st.warning("No data available for visualization.")
        return
        
    try:
        # Define color palette and order (ATUALIZADA)
        group_palette = {
            "Manure-Based": "#1f77b4",          # Azul
            "Coffee Waste": "#d62728",           # Vermelho
            "Agro-Industrial Waste": "#2ca02c", # Verde
            "Plant Waste": "#9467bd",            # Roxo
            "Cellulosic Waste": "#8c564b"        # Marrom
        }
        
        group_order = [
            "Manure-Based",
            "Coffee Waste",
            "Agro-Industrial Waste",
            "Plant Waste",
            "Cellulosic Waste"
        ]
        
        # Filter only groups present in data
        present_groups = [g for g in group_order if g in plot_df[group_var_name].unique()]
        
        min_y_overall = plot_df[dependent_var_name].min()
        max_y_overall = plot_df[dependent_var_name].max()
        
        # Calculate data range for positioning
        data_range = max_y_overall - min_y_overall
        if data_range == 0:  # Avoid division by zero
            data_range = max_y_overall if max_y_overall != 0 else 1
        
        plot_df_max_y = pd.DataFrame()
        if cld_letters:
            # Get max value per group
            max_vals_per_group = plot_df.groupby(group_var_name)[dependent_var_name].max().reset_index()
            
            # Create DF with CLD letters
            cld_df = pd.DataFrame(list(cld_letters.items()), columns=[group_var_name, 'cld_letter'])
            
            # Merge with max values
            plot_df_max_y = max_vals_per_group.merge(cld_df, on=group_var_name)
            
            # Calculate y position with buffer based on data range
            buffer_val = data_range * 0.12
            plot_df_max_y['y_pos'] = plot_df_max_y[dependent_var_name] + buffer_val
            
            # Special adjustment for C/N Ratio
            if "C_N_Ratio" in dependent_var_name:
                plot_df_max_y['y_pos'] = plot_df_max_y[dependent_var_name] * 1.15
                
            plot_df_max_y = plot_df_max_y.dropna(subset=['y_pos', 'cld_letter']).copy()

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Boxplot
        sns.boxplot(
            data=plot_df, 
            x=group_var_name, 
            y=dependent_var_name, 
            ax=ax,
            order=present_groups,
            palette=group_palette,
            width=0.7,
            showfliers=False
        )
        
        # Jitter plot
        sns.stripplot(
            data=plot_df, 
            x=group_var_name, 
            y=dependent_var_name,
            ax=ax,
            order=present_groups,
            palette=group_palette,
            size=7,
            jitter=0.25,
            linewidth=1,
            edgecolor='gray',
            alpha=0.8
        )

        # Add sample size annotations (below the minimum)
        for idx, group in enumerate(present_groups):
            n = sum(plot_df[group_var_name] == group)
            y_pos_n = min_y_overall - (data_range * 0.05)
            ax.text(
                idx, y_pos_n, 
                f"n={n}", 
                ha='center', 
                va='top',
                fontsize=10,
                color='gray'
            )
        
        # Add CLD letters
        if not plot_df_max_y.empty:
            for idx, group in enumerate(present_groups):
                if group in plot_df_max_y[group_var_name].values:
                    group_data = plot_df_max_y[plot_df_max_y[group_var_name] == group]
                    y_pos = group_data['y_pos'].values[0]
                    letter = group_data['cld_letter'].values[0]
                    
                    ax.text(
                        x=idx,
                        y=y_pos,
                        s=letter,
                        ha='center',
                        va='bottom',
                        fontsize=14,
                        color='black',
                        weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
                    )

        # Formatting
        ax.set_title(
            f"Distribution of {dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', '')} by Waste Type",
            fontsize=18,
            pad=20
        )
        ax.set_xlabel("Waste Type Category", fontsize=14, labelpad=15)
        ax.set_ylabel(
            dependent_var_name.replace('_', ' ').replace('perc', '%').replace('final', ''),
            fontsize=14,
            labelpad=15
        )
        
        # Rotate x-axis labels
        plt.xticks(rotation=25, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add grid for readability
        plt.grid(axis='y', alpha=0.3)
        
        # Adjust y-axis limits dynamically
        y_lower_limit = min_y_overall * 0.85
        if min_y_overall > 0:
            y_lower_limit = max(0, min_y_overall * 0.9)
            
        y_upper_limit = max_y_overall * 1.25
        
        # Adjust for CLD letters
        if not plot_df_max_y.empty:
            max_cld_y = plot_df_max_y['y_pos'].max()
            if max_cld_y > y_upper_limit:
                y_upper_limit = max_cld_y * 1.1
                
        plt.ylim(y_lower_limit, y_upper_limit)
        
        # Improve layout
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")

    # --- Interpretation of Results ---
    with st.expander(f"✨ Detailed Interpretation"):
        st.markdown("The interpretation considers tests for homogeneity of variance and normality, and then the results of ANOVA or Kruskal-Wallis and their respective post-hoc tests.")
        
        st.markdown("##### Homogeneity of Variances (Levene's Test):")
        st.markdown(f"- {'Variances ARE homogeneous (p >= 0.05)' if homogeneous_variances else 'Variances are NOT homogeneous (p < 0.05)'}")
        
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
            st.markdown("**Note**: The compact letter display (CLD) summarizes the post-hoc test results. Groups not sharing any letter are significantly different.")
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

# --- Data Loading with Corrections ---
try:
    df = pd.read_excel('dados_vermicomposto_v6.xlsx', sheet_name='Planilha1')
    
    # Renomear colunas para padrão em inglês
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
    
    # Converter colunas numéricas
    numeric_cols = ['N_perc', 'P_perc', 'K_perc', 'pH_final', 'C_N_Ratio_final', 'Duration_days']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # DEPURAÇÃO: Mostrar dados brutos
    st.subheader("Dados Brutos")
    st.dataframe(df.head())
    
except FileNotFoundError:
    st.error("Error: 'dados_vermicomposto_v6.xlsx' file not found. Please ensure it's in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# --- Apply Material Group Categorization (ATUALIZADA) ---
df['Material_Group'] = df['Source_Material'].apply(assign_material_group)

# --- DEPURAÇÃO: Mostrar dados categorizados ---
st.subheader("Dados Categorizados")
st.dataframe(df[['Source_Material', 'Material_Group', 'N_perc', 'P_perc', 'K_perc', 'pH_final', 'C_N_Ratio_final']])

# --- Display Waste Types by Group ---
st.markdown("---")
st.subheader("🔍 Waste Types by Category")
with st.expander("View detailed waste type categorization"):
    if 'Article' in df.columns:
        grouped = df.groupby('Material_Group')
        for name, group in grouped:
            unique_combos = group[['Source_Material', 'Article']].drop_duplicates()
            if not unique_combos.empty:
                st.markdown(f"**{name}**")
                st.markdown(f"*{get_category_description(name)}*")
                for _, row in unique_combos.iterrows():
                    st.markdown(f"- `{row['Source_Material']}` (Source: {row['Article']})")
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
    "Select nutrient to analyze:",
    list(numerical_variables.keys())
)

# --- Execute Analysis ---
run_statistical_analysis_and_plot(df, numerical_variables[selected_variable], "Material_Group")
