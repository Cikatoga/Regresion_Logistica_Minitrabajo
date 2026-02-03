import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar datos
df = pd.read_csv('Churn_Modelling_Limpio.csv')

# 2. Resumen estadístico para variables numéricas (Distribución y Escalas)
print("--- Resumen Estadístico (Escalas y Tendencia Central) ---")
print(df.describe())

# 3. Visualización de Distribuciones y Outliers
# Vamos a analizar las 3 variables numéricas más importantes
variables_num = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary']

plt.figure(figsize=(15, 10))

for i, col in enumerate(variables_num):
    # Histogramas (Distribución)
    plt.subplot(2, 4, i+1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribución de {col}')
    
    # Boxplots (Valores Extremos / Outliers)
    plt.subplot(2, 4, i+5)
    sns.boxplot(y=df[col], color='salmon')
    plt.title(f'Outliers en {col}')

plt.tight_layout()
plt.show()