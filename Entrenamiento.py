import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Cargar y preparar (Repetimos pasos rápidos para que el script sea independiente)
df = pd.read_csv('Churn_Modelling_Limpio.csv')
X = df.drop('Exited', axis=1)
y = df['Exited']

# 2. División de datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Escalado de variables (Vital para la convergencia)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. ENTRENAMIENTO DEL MODELO
modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_train_scaled, y_train)

print("--- Modelo entrenado con éxito ---")
print(f"Variables utilizadas ({len(X.columns)}): {list(X.columns)}")