import pandas as pd

# 1. Cargar el archivo limpio
df_clean = pd.read_csv('Churn_Modelling_Limpio.csv')

# 2. Conteo
conteo = df_clean['Exited'].value_counts()
porcentaje = df_clean['Exited'].value_counts(normalize=True) * 100

print("--- Análisis de Balanceo de Clases ---")
print(f"Conteo por clase:\n{conteo}")
print(f"\nPorcentaje por clase:\n{porcentaje}")

# 3. Tip para el reporte: Ver la diferencia visual
import matplotlib.pyplot as plt
df_clean['Exited'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de Clientes (0: Se quedan, 1: Se van)')
plt.show()