import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 1. CARGAMOS LOS DATOS (Asegúrate de que el nombre del archivo sea el correcto)
df = pd.read_csv('Churn_Modelling_Limpio.csv')

# 2. DEFINIMOS X (variables explicativas) e y (objetivo)
# Usamos las 11 variables que mencionaste
X = df[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain']]
y = df['Exited']

# 3. DIVIDIMOS Y ESCALAMOS (Pasos obligatorios para que el modelo funcione)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. DEFINIMOS Y ENTRENAMOS EL MODELO (Aquí es donde se define 'modelo_log')
modelo_log = LogisticRegression(max_iter=1000)
modelo_log.fit(X_train_scaled, y_train)

# 5. AHORA SÍ: MATRIZ DE CONFUSIÓN (Ejercicio 7.2)
y_pred = modelo_log.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Se Queda (0)', 'Se Va (1)'], 
            yticklabels=['Se Queda (0)', 'Se Va (1)'])
plt.xlabel('Predicción del Modelo')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión')
plt.show()

# 6. REPORTE DE MÉTRICAS (Para ver Precisión y Recall)
print(classification_report(y_test, y_pred))