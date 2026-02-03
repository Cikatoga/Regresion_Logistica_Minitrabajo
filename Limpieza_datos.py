import pandas as pd
import numpy as np

# 1. Cargar los datos
df = pd.read_csv('Churn_Modelling.csv')

# 2. Eliminación de columnas irrelevantes
# RowNumber, CustomerId y Surname no aportan valor predictivo al modelo
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Verificamos si hay nulos
print("Valores faltantes por columna:\n", df_clean.isnull().sum()) 

# Gender: Transformamos a 0 y 1
df_clean['Gender'] = df_clean['Gender'].map({'Female': 0, 'Male': 1})

# Geography: Como tiene 3 categorías (France, Germany, Spain), usamos One-Hot Encoding
# Esto evita darle un orden artificial a los países
df_clean = pd.get_dummies(df_clean, columns=['Geography'], drop_first=True)

# 5. Visualización rápida de la limpieza
print("\nPrimeras filas del dataset limpio:")
print(df_clean.head())

# Guardar el dataset limpio en un nuevo archivo CSV
df_clean.to_csv('Churn_Modelling_Limpio.csv', index=False)

print("¡Archivo guardado con éxito como 'Churn_Modelling_Limpio.csv'!")