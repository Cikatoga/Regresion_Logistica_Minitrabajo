import pandas as pd

# 1. Cargar el dataset limpio que generamos antes
df = pd.read_csv('Churn_Modelling_Limpio.csv')

# 2. Identificación de Outliers en la variable 'Age' (Ejercicio 5.2)
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

outliers_age = df[df['Age'] > limite_superior]

print(f"--- Análisis de Outliers ---")
print(f"Límite superior para Edad: {limite_superior}")
print(f"Cantidad de clientes considerados outliers: {len(outliers_age)}")
print(f"Porcentaje de outliers: {len(outliers_age) / len(df) * 100:.2f}%")

# 3. Ver una muestra de esos outliers
print("\nMuestra de clientes con edad atípica:")
print(outliers_age[['Age', 'Exited']].head())