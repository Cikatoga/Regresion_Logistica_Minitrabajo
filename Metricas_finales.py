import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, auc)

# 1. CARGA DE DATOS
# Asegúrate de que el nombre del archivo coincida con el tuyo
df = pd.read_csv('Churn_Modelling_Limpio.csv')

# 2. SELECCIÓN DE VARIABLES (Las 11 que definimos)
X = df[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain']]
y = df['Exited']

# 3. DIVISIÓN Y ESCALADO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. ENTRENAMIENTO
modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_train_scaled, y_train)

# 5. PREDICCIONES Y PROBABILIDADES
y_pred = modelo_log.predict(X_test_scaled)
y_probs = modelo_log.predict_proba(X_test_scaled)[:, 1]

# 6. CÁLCULO DE MÉTRICAS (Para el Ejercicio 8.2)
print("--- MÉTRICAS DEL MODELO (Umbral 0.5) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print("-" * 40)

# 7. VISUALIZACIÓN: MATRIZ DE CONFUSIÓN (Ejercicio 8.1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Queda (0)', 'Va (1)'], 
            yticklabels=['Queda (0)', 'Va (1)'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')

# 8. VISUALIZACIÓN: CURVA ROC (Ejercicio 8.2)
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
plt.title('Curva ROC')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

print(f"Cálculo del AUC: {roc_auc:.4f}")