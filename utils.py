import hashlib
import math
import re

def is_hash(senha):
    return all(c in '0123456789abcdefABCDEF' for c in senha) and len(senha) in [32, 40, 64]

#Verifica se a string passada parece ser um hash (ex: MD5, SHA1 ou SHA256).

def calcular_entropia(senha):
    pool = 0
    if re.search(r'[a-z]', senha): pool += 26
    if re.search(r'[A-Z]', senha): pool += 26
    if re.search(r'[0-9]', senha): pool += 10
    if re.search(r'[^a-zA-Z0-9]', senha): pool += 32
    if pool == 0: return 0
    return round(len(senha) * math.log2(pool), 2)

#Calcula a entropia da senha — uma medida matemática de complexidade e imprevisibilidade.
#Senhas com maior entropia tendem a ser mais fortes. Essa métrica é uma feature importante para o modelo de Machine Learning prever a força da senha.
#Exemplo:
#123456 → entropia baixa
#S3nh@S3gura! → entropia alta


def extrair_features(senha, rockyou):
    if not isinstance(senha, str):
        senha = str(senha) if senha is not None else ""

    comprimento = len(senha)
    qtd_numeros = sum(c.isdigit() for c in senha)
    qtd_maiusculas = sum(c.isupper() for c in senha)
    qtd_simbolos = sum(not c.isalnum() for c in senha)
    apareceu_no_rockyou = int(senha in rockyou)

    # Entropia
    if len(senha) > 0:
        alfabeto = set(senha)
        probas = [senha.count(c)/len(senha) for c in alfabeto]
        entropia = -sum(p * math.log2(p) for p in probas)
    else:
        entropia = 0

    return {
        "senha": senha,
        "comprimento": comprimento,
        "qtd_numeros": qtd_numeros,
        "qtd_maiusculas": qtd_maiusculas,
        "qtd_simbolos": qtd_simbolos,
        "entropia": entropia,
        "apareceu_no_rockyou": apareceu_no_rockyou
    }


#Extrai um conjunto de características (features) da senha, que serão usadas para treinar ou prever no modelo ML.
#As features são:

#senha: o texto original (para aplicar TF-IDF)
#comprimento: tamanho da senha
#qtd_numeros: quantidade de dígitos
#qtd_maiusculas: quantidade de letras maiúsculas
#qtd_simbolos: quantidade de símbolos
#entropia: complexidade da senha
#apareceu_no_rockyou: 0 ou 1, indicando se a senha foi encontrada em um vazamento (indicador de fragilidade)#