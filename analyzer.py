import pandas as pd
import joblib
from utils import extrair_features
import os

# Carregar senhas
with open("data/input_passwords.txt", encoding="utf-8") as f:
    senhas = [line.strip() for line in f]

with open("data/rockyou.txt", encoding="latin-1") as f:
    rockyou = set(line.strip() for line in f)

#Carrega as senhas que o usuário quer analisar (uma por linha);
#Carrega a wordlist rockyou.txt para verificar se a senha já é conhecida como fraca.

# Extrair features
dados = [extrair_features(s, rockyou) for s in senhas]
df = pd.DataFrame(dados)

#Gera um dicionário com os dados de cada senha;
#Converte a lista de dicionários em um DataFrame para alimentar o modelo.

# Carregar modelo treinado
modelo = joblib.load("modelo_senhas.joblib")

# Prever força
predicoes = modelo.predict(df)
df["forca_predita"] = predicoes

#Carrega o modelo treinado salvo no ml_model.py;
#Aplica a predição no novo DataFrame com as senhas do usuário;
#Adiciona uma nova coluna forca_predita com o resultado do modelo.

os.makedirs("output", exist_ok=True)

# Salvar relatório
df.to_csv("output/relatorio.csv", index=False)
print("Relatório salvo em 'output/relatorio.csv'")

#Salva os dados finais (incluindo a força prevista) em um arquivo .csv no diretório output/.
