import streamlit as st
import pandas as pd
import joblib
from utils import extrair_features
import matplotlib.pyplot as plt
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
    senhas = [line.strip() for line in uploaded_file.readlines() if line.strip()]
    
    # Extrair features
    dados = [extrair_features(s, rockyou) for s in senhas]
    df = pd.DataFrame(dados)
    
    # Predizer força
    predicoes = modelo.predict(df)
    df["forca_predita"] = predicoes

    # Mostrar DataFrame
    st.subheader("📋 Resultado da análise")
    st.dataframe(df)

    # Gráfico
    st.subheader("📊 Distribuição da força das senhas")
    fig, ax = plt.subplots()
    df["forca_predita"].value_counts().plot(kind="bar", ax=ax, color=["red", "orange", "green"])
    plt.xlabel("Força")
    plt.ylabel("Quantidade")
    plt.title("Distribuição das Senhas")
    st.pyplot(fig)

    # Botão para baixar CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar relatório em CSV", csv, "relatorio.csv", "text/csv")
