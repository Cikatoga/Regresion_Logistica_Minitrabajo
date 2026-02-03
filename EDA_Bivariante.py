import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_csv('Churn_Modelling_Limpio.csv')

# --- 3.2 Relación Variable Explicativa vs Objetivo ---

# A. Relación Edad vs Abandono (Variable Continua vs Categórica)
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x='Age', hue='Exited', fill=True, palette='magma')
plt.title('Distribución de Edad por Estado de Abandono')
plt.show()

# B. Relación Geografía vs Abandono (Variable Categórica vs Categórica)
# Nota: Si ya hiciste get_dummies, usa el df original o reconstruye para la tabla
tab_geo = pd.crosstab(df['Geography_Germany'], df['Exited'], normalize='index') * 100
print("Tabla Cruzada: Alemania vs Abandono (%)")
print(tab_geo)

# --- 3.2 Multicolinealidad (Relación entre explicativas) ---

# Calculamos la matriz de correlación
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación (Detección de Multicolinealidad)')
plt.show()