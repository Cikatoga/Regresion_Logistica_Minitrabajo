import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 1. Cargar datos
df = pd.read_csv('Churn_Modelling_Limpio.csv')

# 2. Variables (Las 11 que ya confirmaste que usas)
X = df[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain']]
y = df['Exited']

# 3. Preparación (Dividir y Escalar)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entrenamiento
modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_train_scaled, y_train)

# 5. Obtener Probabilidades (necesario para cambiar el umbral)
probs = modelo_log.predict_proba(X_test_scaled)[:, 1]

# --- COMPARACIÓN DE UMBRALES ---

umbrales = [0.5, 0.3]
plt.figure(figsize=(14, 5))

for i, u in enumerate(umbrales):
    # Aplicar umbral
    y_pred_u = (probs >= u).astype(int)
    cm = confusion_matrix(y_test, y_pred_u)
    
    # Graficar
    plt.subplot(1, 2, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Queda', 'Va'], yticklabels=['Queda', 'Va'])
    plt.title(f'Matriz de Confusión (Umbral {u})')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')

plt.tight_layout()
plt.show()

# 6. Reporte detallado para el informe (Umbral 0.5)
print("--- REPORTE DETALLADO (UMBRAL 0.5) ---")
print(classification_report(y_test, (probs >= 0.5).astype(int)))

print("\n--- REPORTE DETALLADO (UMBRAL 0.3) ---")
print(classification_report(y_test, (probs >= 0.3).astype(int)))