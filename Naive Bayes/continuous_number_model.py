from sklearn.datasets import make_classification # gerar dados de classificação
from sklearn.model_selection import train_test_split # separar treino e teste
from sklearn.naive_bayes import GaussianNB # modelo Naive Bayes para dados contínuos
from sklearn.metrics import accuracy_score, f1_score, classification_report # avaliar o modelo
import pandas as pd


# Criar dataset 
df = pd.read_csv("data2.csv")

# Separar atributos (X) e classe (y)
X = df.drop("play", axis=1)
y = df["play"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Fazer previsões
y_prev = model.predict(X_test)

# Avaliar
acc = accuracy_score(y_test, y_prev)
f1 = f1_score(y_test, y_prev, average="weighted")  # <-- calcular F1-score
print("F1-score do modelo:", f1)
print("\nAcurácia do modelo:", acc)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_prev))

# Interpretação dos resultados
"""
Interpretação dos Resultados:

- Classe 0:
    Precision: 0.70 → Das previsões feitas como classe 0, 70% estavam corretas.
    Recall: 1.00 → Detectou 100% dos exemplos reais da classe 0.
    F1-score: 0.82 → Bom equilíbrio entre precisão e recall para classe 0.

- Classe 1:
    Precision: 1.00 → Todas as previsões feitas como classe 1 estavam corretas.
    Recall: 0.50 → Detectou apenas 50% dos exemplos reais da classe 1.
    F1-score: 0.67 → Baixo recall derrubou o valor geral.

- Accuracy: 0.77 → O modelo acertou 77% das previsões no total.
- Macro avg: média simples das métricas das classes (ignora desequilíbrio de classe).
- Weighted avg: média ponderada pelo número de exemplos de cada classe.

Definição simples das métricas:

- Precision (Precisão): proporção de previsões corretas em relação ao total de previsões feitas para uma classe.
- Recall (Sensibilidade): proporção de exemplos corretos detectados em relação ao total de exemplos reais daquela classe.
- F1-score: média harmônica entre precisão e recall, balanceando os dois.
- Accuracy (Acurácia): proporção geral de previsões corretas.
- Macro avg: média simples das métricas por classe.
- Weighted avg: média ponderada das métricas por classe, considerando o número de exemplos.
"""
