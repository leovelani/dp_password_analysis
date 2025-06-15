# Como rodar:
```
pip install -r requirements.txt
```
- Baixar rockyou.txt: https://www.kaggle.com/datasets/wjburns/common-password-list-rockyoutxt
- Inserir o txt na pasta "data"

```
python ml_model.py
```

```
python analyzer.py
```

#  Analisador de Força de Senhas

Este projeto tem como objetivo desenvolver uma solução automatizada capaz de **classificar a força de senhas**, utilizando técnicas de **aprendizado de máquina**, análise heurística e verificação contra bases de dados de senhas vazadas (como a wordlist `rockyou.txt`).

O sistema é capaz de receber como entrada um conjunto de senhas (em texto claro ou hashes) e, com base em suas características e padrões, **identificar se cada senha é fraca, média ou forte**. A ferramenta pode ser usada para análises de segurança, auditorias de sistemas, validação de políticas de senhas e testes de robustez.

---

##  Componentes principais

- **`utils.py`**: Contém funções auxiliares para calcular entropia, detectar senhas presentes em vazamentos, e extrair características das senhas.
- **`ml_model.py`**: Responsável por gerar um dataset com rótulos automáticos (fraca, média, forte) e treinar um modelo de aprendizado de máquina com base nessas informações.
- **`analyzer.py`**: Aplica o modelo treinado para classificar novas senhas e gera um relatório final em CSV com os resultados.
- **`rockyou.txt`**: Wordlist utilizada para detectar senhas comuns/vazadas.
- **`input_passwords.txt`**: Arquivo de entrada com senhas a serem analisadas.
- **`output/relatorio.csv`**: Arquivo de saída com os resultados das classificações.

---

##  Utilização do algoritmo **Random Forest**

Para realizar a classificação automática da força das senhas, o projeto utiliza o algoritmo **Random Forest**, uma técnica de **aprendizado supervisionado baseada em conjunto de árvores de decisão**.

### Por que Random Forest?

- Ele é **robusto contra overfitting**, pois combina os resultados de diversas árvores de decisão para chegar a uma resposta mais estável.
- Lida muito bem com diferentes tipos de dados, como:
  - Texto transformado em vetores (via TF-IDF dos caracteres da senha);
  - Dados numéricos (como comprimento, entropia, quantidade de dígitos e símbolos);
  - Dados categóricos (como presença ou não em vazamentos públicos).
- Permite detectar **padrões complexos entre as variáveis** de forma mais eficaz do que algoritmos lineares ou probabilísticos simples (como o Naive Bayes).

### Como ele é aplicado neste projeto?

Durante o treinamento:

- As senhas são convertidas em um conjunto de **features numéricas** (como entropia, comprimento, uso de símbolos) e **vetores TF-IDF** baseados nos caracteres da senha.
- Com base nessas features, o Random Forest aprende a reconhecer padrões associados a senhas fracas, médias e fortes.
- Os rótulos de força são definidos heurística e automaticamente para possibilitar o treinamento supervisionado.

Após o modelo ser treinado, ele é salvo e reutilizado no `analyzer.py` para classificar qualquer nova senha inserida pelo usuário.

---

##  Exemplo de saída

Após a execução, o projeto gera um relatório `.csv` com as colunas:

```csv
senha,comprimento,qtd_numeros,qtd_maiusculas,qtd_simbolos,entropia,apareceu_no_rockyou,forca_predita
123456,6,6,0,0,19.93,1,fraca
Senha123,9,3,1,0,42.51,0,média
Sup3r$3gur@2024,15,4,2,3,77.47,0,forte
