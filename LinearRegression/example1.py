import numpy as np

x = np.array([20, 30, 40]) # idades
y = np.array([2000, 3000, 4000]) # salários

# medias
mx = np.mean(x)
my = np.mean(y)

# desvio padrão -> quantidade de variação em relação a média
sx = np.std(x)
sy = np.std(y)

# Correlação de Pearson ->  o quanto duas variáveis estão relacionadas
r = np.corrcoef(x, y)[0, 1] # [0, 1] -> Pega o valor da correlação entre x e y

# Desvio padrão amostral -> O quanto os dados estão dispersos (quão longe estão da média)
ssx = np.std(x, ddof=1)
ssy = np.std(y, ddof=1)

# Inclinação -> O quanto y cresce para cada unidade que x cresce
b1 = r * (sy / sx)
# Intercepto -> O valor de y quando x é 0
b0 = my - (b1 * mx)
# Equação da regressão linear
def regressao_linear(x):
    return b0 + b1 * x
  
# Teste
input = int(input("Digite a idade: "))
print(round(regressao_linear(input)))
