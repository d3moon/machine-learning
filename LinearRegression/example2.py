import numpy as np

class LinearRegression:
    def __init__(self, x, y):
      # Inicializa os dados e calcula os parâmetros da regressão
        self.x = np.array(x) # idades
        self.y = np.array(y) # salários
        self.mx = np.mean(self.x) # média de x
        self.my = np.mean(self.y) # média de y
        self.sx = np.std(self.x) # desvio padrão de x
        self.sy = np.std(self.y) # desvio padrão de y
        self.r = np.corrcoef(self.x, self.y)[0, 1] # correlação de Pearson
        self.b1 = self.r * (self.sy / self.sx) # inclinação
        self.b0 = self.my - (self.b1 * self.mx) # intercepto
        
    def predict(self, x):
        return self.b0 + self.b1 * x # Equação da regressão linear


# Exemplo de uso
x = [20, 30, 40]  # idades
y = [2000, 3000, 4000]  # salários
model = LinearRegression(x, y)
input_age = int(input("Digite a idade: "))
predicted_salary = model.predict(input_age)
print(f"Salário previsto para a idade {input_age}: {round(predicted_salary)}")