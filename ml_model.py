import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from utils import extrair_features
import re

def gerar_dataset(senhas, rockyou):
    dados = [extrair_features(s, rockyou) for s in senhas]
    df = pd.DataFrame(dados)
    
    def padrao_obvio(senha):
        # Padrões comuns e repetições
        padroes = [
            r"123456", r"abcdef", r"qwerty", r"senha", r"password",
            r"(.)\1{2,}",  # repetições de 3 ou mais caracteres
        ]
        for p in padroes:
            if re.search(p, senha, re.IGNORECASE):
                return True
        return False

    def rotulo(row):
        senha = row["senha"]
        comprimento = row["comprimento"]
        qtd_maiusculas = row["qtd_maiusculas"]
        qtd_simbolos = row["qtd_simbolos"]
        qtd_numeros = row["qtd_numeros"]
        entropia = row["entropia"]
        apareceu_no_rockyou = row["apareceu_no_rockyou"]
        tipos = sum([
            any(c.islower() for c in senha),
            any(c.isupper() for c in senha),
            any(c.isdigit() for c in senha),
            any(not c.isalnum() for c in senha)
        ])

        # Fraca
        if (
            apareceu_no_rockyou or
            comprimento < 8 or
            tipos < 2 or
            padrao_obvio(senha) or
            entropia < 28
        ):
            return "fraca"
        # Forte
        elif (
            comprimento >= 12 and
            tipos == 4 and
            qtd_simbolos > 0 and
            entropia > 40 and
            not apareceu_no_rockyou and
            not padrao_obvio(senha)
        ):
            return "forte"
        # Média
        else:
            return "média"
    
    df["forca"] = df.apply(rotulo, axis=1)
    return df
#
#Gera um dataframe com features das senhas;
#Aplica rótulos heurísticos (forte, média, fraca) para treinar o modelo:
#Se está no rockyou.txt → fraca
#Se tem ≥12 caracteres e símbolo → forte
#Caso contrário → média

def treinar_modelo(df):
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
    colunas_numericas = ["comprimento", "qtd_numeros", "qtd_maiusculas", "qtd_simbolos", "entropia", "apareceu_no_rockyou"]
    
    preprocessor = ColumnTransformer([
        ("tfidf", tfidf, "senha"),
        ("num", StandardScaler(), colunas_numericas)
    ])
    
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X = df.drop("forca", axis=1)
    y = df["forca"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    joblib.dump(pipeline, "modelo_senhas.joblib")
    print("Modelo treinado e salvo como 'modelo_senhas.joblib'")

#Define o pipeline de ML:

#TfidfVectorizer: extrai n-gramas de caracteres da senha;
#StandardScaler: normaliza as features numéricas;
#RandomForestClassifier: classificador principal.
#Divide os dados em treino e teste;
#Treina o modelo e salva como arquivo .joblib.

if __name__ == "__main__":
    with open("data/input_passwords.txt", encoding="utf-8") as f:
        senhas = [line.strip() for line in f]

    with open("data/rockyou.txt", encoding="latin-1") as f:
        rockyou = set(line.strip() for line in f)

    df = gerar_dataset(senhas, rockyou)
    treinar_modelo(df)

#Carrega as senhas e a wordlist rockyou.txt;
#Gera o dataset rotulado com gerar_dataset;
#Treina e salva o modelo com treinar_modelo.