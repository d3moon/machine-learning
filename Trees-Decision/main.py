import pandas as pd                  # Para manipular dados (tabelas)
from sklearn.model_selection import train_test_split  # Para dividir treino/teste
from sklearn.tree import DecisionTreeClassifier, plot_tree  # O modelo e o visualizador
from sklearn.metrics import accuracy_score             # Para medir desempenho
import matplotlib.pyplot as plt                        # Para gráficos

# Criando o conjunto de dados
data = {
    'Tempo': ['Sol', 'Sol', 'Nublado', 'Chuva', 'Chuva', 'Chuva', 'Nublado', 'Sol', 'Sol', 'Chuva', 'Sol', 'Nublado', 'Nublado', 'Chuva'],
    'Temperatura': ['Quente', 'Quente', 'Quente', 'Amena', 'Fria', 'Fria', 'Fria', 'Amena', 'Fria', 'Amena', 'Amena', 'Amena', 'Quente', 'Amena'],
    'Umidade': ['Alta', 'Alta', 'Alta', 'Alta', 'Normal', 'Normal', 'Normal', 'Alta', 'Normal', 'Normal', 'Normal', 'Alta', 'Normal', 'Alta'],
    'Vento': ['Fraco', 'Forte', 'Fraco', 'Fraco', 'Fraco', 'Forte', 'Forte', 'Fraco', 'Fraco', 'Fraco', 'Forte', 'Forte', 'Fraco', 'Forte'],
    'Jogar': ['Não', 'Não', 'Sim', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Sim', 'Sim', 'Sim', 'Não']
}

# Transformando em DataFrame (tabela do Pandas)
df = pd.DataFrame(data)
print(df)

# Convertendo categorias para números

"""
.astype('category') → transforma o texto em categorias.
.cat.codes → converte cada categoria em um número (ex: “Sol” → 2, “Chuva” → 0, etc).
"""
for col in ['Tempo', 'Temperatura', 'Umidade', 'Vento', 'Jogar']:
    df[col] = df[col].astype('category').cat.codes

print(df)

# Separando as variáveis
X = df.drop('Jogar', axis=1)  # Atributos
y = df['Jogar']               # Rótulo (o que queremos prever)

# 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando a árvore (usando o critério 'entropy' para medir o ganho de informação)
model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Treinando o modelo
model.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando a precisão
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc*100:.2f}%")

plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=['Não', 'Sim'], filled=True)
plt.show()






