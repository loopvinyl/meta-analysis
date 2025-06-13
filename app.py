import pandas as pd
import streamlit as st
import altair as alt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import numpy as np

# --- 1. Carregamento de Dados ---
# O arquivo dados_vermicomposto.xlsx deve estar na mesma pasta que este script.
try:
    df = pd.read_excel('dados_vermicomposto.xlsx')
except FileNotFoundError:
    st.error("Erro: Arquivo 'dados_vermicomposto.xlsx' não encontrado. Por favor, certifique-se de que ele está na mesma pasta que este script.")
    st.stop()

# NOVO CÓDIGO AQUI: Renomear a coluna 'Material de Origem do Vermicomposto' para 'Material_Group'
if 'Material de Origem do Vermicomposto' in df.columns:
    df.rename(columns={'Material de Origem do Vermicomposto': 'Material_Group'}, inplace=True)
else:
    st.error("Erro: Coluna 'Material de Origem do Vermicomposto' não encontrada no arquivo de dados. Por favor, verifique o nome da coluna no seu Excel.")
    st.stop()

# Converte 'Material_Group' para categoria para garantir tratamento correto
if 'Material_Group' in df.columns: # Agora 'Material_Group' DEVE existir
    df['Material_Group'] = df['Material_Group'].astype('category')
else:
    # Esta mensagem de erro só deve aparecer se o renomeio acima falhou por algum motivo inesperado
    st.error("Erro interno: 'Material_Group' não foi criada corretamente. Verifique o código ou o Excel.")
    st.stop()

# --- 2. Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Metanálise de Vermicompostos",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Análise Metanalítica de Vermicompostos por Tipo de Resíduo")
st.markdown("Este aplicativo interativo permite explorar os resultados dos testes estatísticos de Nitrogênio, Fósforo, Potássio, pH e Razão C/N em vermicompostos, agrupados por material de origem.")
st.markdown("---")

# ... O RESTO DO CÓDIGO DO app.py PERMANECE O MESMO ...
# (As seleções de variáveis e as funções de análise já usam 'Material_Group')
