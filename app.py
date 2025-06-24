import streamlit as st
import pandas as pd

st.set_page_config(page_title="Teste de Diagnóstico", layout="centered")
st.title("🔍 Diagnóstico do Streamlit")

st.write("📌 Etapa 1: Início do app carregado com sucesso.")

try:
    st.write("📌 Etapa 2: Tentando ler o Excel...")
    df = pd.read_excel("dados_vermicomposto_v6.xlsx", sheet_name="Planilha1")
    st.success("✅ Excel carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Falha ao carregar o Excel: {e}")
    st.stop()

st.write("📌 Etapa 3: Visualizando preview dos dados:")
st.dataframe(df.head())

st.write("📌 Etapa 4: Fim do teste.")
