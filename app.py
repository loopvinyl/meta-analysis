import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import io

st.set_page_config(page_title="Análise de Variância com Tukey", layout="wide")

st.title("📊 Análise de Variância (ANOVA) + Teste de Tukey com Letras")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Faça upload de um arquivo CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Arquivo carregado com sucesso!")
        st.write("Pré-visualização dos dados:")
        st.dataframe(df)

        # Seleção das variáveis
        cols = df.columns.tolist()
        group_col = st.selectbox("Selecione a variável de grupo (fator)", cols)
        value_col = st.selectbox("Selecione a variável de resposta", cols)

        if group_col != value_col:
            # ANOVA
            st.subheader("📈 Resultados da ANOVA")
            model = ols(f"{value_col} ~ C({group_col})", data=df).fit()
            anova_table = anova_lm(model, typ=2)
            st.write(anova_table)

            # Teste de Tukey
            st.subheader("🧪 Teste de Tukey HSD")
            tukey = pairwise_tukeyhsd(df[value_col], df[group_col])
            st.write(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))

            # Letras de significância
            mc = MultiComparison(df[value_col], df[group_col])
            result = mc.tukeyhsd()
            letters = mc.groupsunique
            cld = mc.tukeyhsd().groupsunique

            # Geração das letras
            from statsmodels.stats.multicomp import MultiComparison
            import statsmodels.stats.multicomp as mc_lib

            comps = MultiComparison(df[value_col], df[group_col])
            res = comps.tukeyhsd()
            groups_df = pd.DataFrame(data={group_col: comps.groupsunique})
            groups_df['letras'] = mc_lib.tukeyhsd(df[value_col], df[group_col]).groupsunique

            # Melhor: usar função get_letters para gerar letras reais
            from statsmodels.sandbox.stats.multicomp import get_tukey_pvalue, get_tukeyhsd_summary, get_tukey_letters

            summary_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            letters_df = get_tukey_letters(summary_df)
            letter_map = letters_df.set_index("group")["letters"].to_dict()

            # Plot com letras
            st.subheader("📉 Boxplot com Letras de Significância")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x=group_col, y=value_col, ax=ax, palette="pastel")

            # Adicionar letras ao gráfico
            group_means = df.groupby(group_col)[value_col].mean()
            for i, (group, mean) in enumerate(group_means.items()):
                letter = letter_map.get(group, "")
                ax.text(i, mean + (df[value_col].max() * 0.02), letter,
                        ha='center', va='bottom', fontsize=14, color='black')

            st.pyplot(fig)

        else:
            st.warning("A variável de grupo e a variável de resposta devem ser diferentes.")

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")

else:
    st.info("Aguardando upload de um arquivo CSV.")
