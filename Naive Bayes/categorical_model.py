import pandas as pd # manipular tabela (DataFrame).
from sklearn.model_selection import train_test_split # separar treino e teste
from sklearn.naive_bayes import CategoricalNB # modelo Naive Bayes para dados categóricos
from sklearn.preprocessing import LabelEncoder # transformar strings em números
from sklearn.metrics import accuracy_score, f1_score # avaliar o modelo
# --- Carregar os dados ---
data = pd.read_csv("data.csv")

# Mostrar as primeiras linhas
df = pd.DataFrame(data)
print(df.head())

# --- Pré-processamento: transformar strings em números ---
encoders = {} # dicionário para armazenar os LabelEncoders
for col in df.columns: # para cada coluna
    le = LabelEncoder() # criar um LabelEncoder
    df[col] = le.fit_transform(df[col]) # transformar a coluna em números
    encoders[col] = le # armazenar o encoder

print("\nDados codificados:")
print(df.head())

# --- Separar atributos (X) e classe (y) ---
X = df.drop("play", axis=1)
y = df["play"]

# --- Separar treino e teste ---
# Ele pega 30% dos dados para teste e o resto para treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # random_state é usado para controlar a aleatoriedade, garantindo que resultados sejam previsíveis e reproduzíveis.

# --- Criar e treinar o modelo Naive Bayes ---
# Naive Bayes é um algoritmo que realiza classificações baseadas no Teorema de Bayes (probabilidades condicionais).
model = CategoricalNB()
model.fit(X_train, y_train)

# --- Fazer previsões ---
y_pred = model.predict(X_test)

print("\nPrevisões:", y_pred)
print("Valores reais:", list(y_test))

# --- Avaliar ---
acc = accuracy_score(y_test, y_pred) # acurácia: resultado -> proporção de acertos em relação ao total
f1 = f1_score(y_test, y_pred, average="weighted")  # <-- calcular F1-score
print("F1-score do modelo:", f1)
print("\nAcurácia do modelo:", acc)