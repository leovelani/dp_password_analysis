import streamlit as st
import pandas as pd
import joblib
from utils import extrair_features
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregar modelo
modelo = joblib.load("modelo_senhas.joblib")

# Título do app
st.title("🔐 Analisador de Força de Senhas")
st.write("Este dashboard analisa a força de senhas com base em características e um modelo treinado.")

# Upload do arquivo de senhas
uploaded_file = st.file_uploader("Envie um arquivo .txt com uma senha por linha", type=["txt"])

# Carregar wordlist rockyou.txt
with open("data/rockyou.txt", encoding="latin-1") as f:
    rockyou = set(line.strip() for line in f)

if uploaded_file:
    # Ler senhas do arquivo enviado
    senhas = [line.decode("utf-8").strip() for line in uploaded_file]  # ao invés de readlines()
    
    # Extrair feature
    dados = [extrair_features(s, rockyou) for s in senhas]
    df = pd.DataFrame(dados)
    
    # Predizer força
    predicoes = modelo.predict(df)
    df["forca_predita"] = predicoes

    # Mostrar DataFrame
    st.subheader("📋 Resultado da análise")
    def colorir_linhas(row):
        cor = ""
        if row["forca_predita"] == "fraca":
            cor = "background-color: #ffcccc"
        elif row["forca_predita"] == "média":
            cor = "background-color: #fff3cd"
        elif row["forca_predita"] == "forte":
            cor = "background-color: #d4edda"
        return [cor] * len(row)
    st.dataframe(df.style.apply(colorir_linhas, axis=1))


    # Gráfico
    st.subheader("📊 Distribuição da força das senhas")
    fig, ax = plt.subplots()
    df["forca_predita"].value_counts().plot(kind="bar", ax=ax, color=["red", "orange", "green"])
    plt.xlabel("Força")
    plt.ylabel("Quantidade")
    plt.title("Distribuição das Senhas")
    st.pyplot(fig)

    st.subheader("📈 Complexidade das senhas (entropia x comprimento)")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="comprimento",
        y="entropia",
        hue="forca_predita",
        palette={"fraca": "red", "média": "orange", "forte": "green"},
        ax=ax
    )
    ax.set_title("Entropia vs. Comprimento")
    st.pyplot(fig)

    st.subheader("📊 Senhas por quantidade de símbolos e força")
    contagem = df.groupby(["qtd_simbolos", "forca_predita"]).size().unstack().fillna(0)
    fig4, ax4 = plt.subplots()
    contagem.plot(kind="bar", stacked=True, ax=ax4, colormap="viridis")
    plt.xlabel("Quantidade de símbolos")
    plt.ylabel("Nº de senhas")
    plt.title("Distribuição por símbolo e força")
    st.pyplot(fig4)

    # Botão para baixar CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar relatório em CSV", csv, "relatorio.csv", "text/csv")
