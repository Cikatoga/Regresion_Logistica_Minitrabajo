import pandas as pd

# Cargar el dataset
df = pd.read_csv('Churn_Modelling.csv')

# Verificar valores nulos por columna
print("Conteo de valores faltantes:")
print(df.isnull().sum())