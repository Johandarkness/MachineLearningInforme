import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

# Ignorar las advertencias
warnings.filterwarnings("ignore")

# Paso 1: Leer el archivo stroke.csv
datos = pd.read_csv('stroke.csv')

# Paso 2: Seleccionar aleatoriamente el 80% del conjunto de datos para entrenar y el 20% restante para las pruebas
X_train, X_test, y_train, y_test = train_test_split(datos.drop('stroke', axis=1), datos['stroke'], test_size=0.2, random_state=123)

# Paso 3: llenar los datos faltantes y normalizar los datos
categoricas = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numericas = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Codificar variables categóricas usando LabelEncoder
label_encoder = LabelEncoder()
for categoria in categoricas:
    X_train[categoria] = label_encoder.fit_transform(X_train[categoria])
    X_test[categoria] = label_encoder.transform(X_test[categoria])

# Llenar los datos faltantes con la media usando SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train[numericas] = imputer.fit_transform(X_train[numericas])
X_test[numericas] = imputer.transform(X_test[numericas])


# Paso 4: Configurar los hiperparámetros del árbol de decisión (criterion=gini, splitter=best, random_state=123)
hiperparametros = {
    'criterion': 'gini',
    'splitter': 'best',
    'random_state': 123
}

resultados_1 = []

for max_depth in range(5, 51, 5):
    hiperparametros['max_depth'] = max_depth
    modelo = DecisionTreeClassifier(**hiperparametros)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    resultados_1.append((max_depth, accuracy))

# Paso 5: Mostrar una tabla con el accuracy para los 10 árboles
tabla1 = pd.DataFrame(resultados_1, columns=['max_depth', 'Accuracy (criterion=gini)'])
print("Tabla 1: Accuracy para los árboles con criterion=gini")
print(tabla1)

