import pandas as pd
import matplotlib.pyplot as plt # lib para plotagem de gráficos
import seaborn as sns # lib para plotagem de gráficos mais avançados e estilizados

"""
Análise Exploratória de Dados (EDA – Exploratory Data Analysis) utilizando gráficos de heatmap, dispersão e boxplot. 
"""

dados = pd.read_csv('resources/veiculos.csv')

# plotando gráfico de heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(dados.corr(), annot=True, cmap='coolwarm')
plt.show()

# plotando gráfico de dispersão -> relação entre km_rodados e preco. o quanto maior o km_rodados, menor o preco
plt.figure(figsize=(10, 6))
sns.scatterplot(x='km_rodados', y='preco', data=dados)
plt.show()

# plotando gráfico de boxplot -> relação entre preco e km_rodados -> quanto maior o km_rodados, menor o preco
plt.figure(figsize=(10, 6))
sns.boxplot(x='km_rodados', y='preco', data=dados)
plt.show()