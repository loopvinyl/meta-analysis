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
    # (Manter o mesmo código original aqui)
    # ...
    return final_letters

# --- Material Group Categorization Function (ATUALIZADA) ---
def assign_material_group(source):
    if pd.isna(source):
        return "Uncategorized"
    source = str(source).lower()
    
    # 1. Agro-Industrial Waste (Coffee)
    coffee_keywords = ["coffee", "scg", "borra", "café"]
    if any(kw in source for kw in coffee_keywords):
        return "Agro-Industrial Waste (Coffee)"
    
    # 2. Agro-Industrial Waste (Fruit)
    fruit_keywords = ["pineapple", "abacaxi", "banana", "bl", "fruta"]
    if any(kw in source for kw in fruit_keywords):
        return "Agro-Industrial Waste (Fruit)"
    
    # 3. Agro-Industrial Waste (Crop Residues)
    crop_keywords = ["bagasse", "crop residue", "straw", "palha", "sugarcane", "bagaço", "resíduo agrícola"]
    if any(kw in source for kw in crop_keywords):
        return "Agro-Industrial Waste (Crop Residues)"
    
    # 4. Animal Manure-Based
    manure_keywords = ["manure", "dung", "cattle", "cow", "bovine", "cd", "vr", "fezes", "estrume", "gado", "vaca"]
    if any(kw in source for kw in manure_keywords):
        return "Animal Manure-Based"
    
    # 5. Food Waste
    if "food" in source or "kitchen" in source or "alimento" in source:
        return "Food Waste"
    
    # 6. Green Waste
    green_keywords = ["vegetable", "grass", "water hyacinth", "weeds", "parthenium", "green", "verde", "erva", "grama"]
    if any(kw in source for kw in green_keywords):
        return "Green Waste"
    
    # 7. Cellulosic Waste
    cellulosic_keywords = ["cardboard", "paper", "newspaper", "filters", "filtro", "cellulose", "papel", "papelão"]
    if any(kw in source for kw in cellulosic_keywords):
        return "Cellulosic Waste"
    
    return "Uncategorized"

# --- Get category description ---
def get_category_description(category):
    # (Manter o mesmo código original aqui)
    # ...
    return descriptions.get(category, "No description available")

# --- Function to run statistical analysis and display results ---
def run_statistical_analysis_and_plot(data, dependent_var_name, group_var_name):
    # (Manter o mesmo código original aqui)
    # ...

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
    df = pd.read_excel('dados_vermicomposto_v5.xlsx', sheet_name='Planilha1')
    
    # Traduzir valores em português
    df.replace({
        'Não reportado': 'Not reported',
        'Não reportada': 'Not reported',
        'Valores finais': 'Final values',
        'vermicomposto final': 'final vermicompost'
    }, inplace=True)
    
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
    
    # Converter colunas numéricas (tratar valores não numéricos)
    numeric_cols = ['N_perc', 'P_perc', 'K_perc', 'pH_final', 'C_N_Ratio_final']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preencher valores ausentes na duração
    if 'Duration_days' in df.columns:
        df['Duration_days'].fillna(0, inplace=True)
        df['Duration_days'] = df['Duration_days'].astype(int)
        df.loc[df['Duration_days'] == 0, 'Duration_days'] = None
        
except FileNotFoundError:
    st.error("Error: 'dados_vermicomposto_v5.xlsx' file not found. Please ensure it's in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# --- Apply Material Group Categorization ---
df['Material_Group'] = df['Source_Material'].apply(assign_material_group)

# --- Filter out non-vermicompost entries ---
if 'Observations' in df.columns:
    filter_condition = (
        df['Observations'].str.contains('drum compost|not vermicompost', case=False, na=False) |
        df['Source_Material'].str.contains('drum compost', case=False, na=False)
    )
    df = df[~filter_condition]

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
