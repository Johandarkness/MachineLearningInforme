import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
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
print("\n","#####################################################################","\n")

# Paso 6: Configurar los hiperparámetros del árbol de decisión (criterion=entropy, splitter=best, random_state=123)
hiperparametros2 = {
    'criterion': 'entropy',
    'splitter': 'best',
    'random_state': 123
}

resultados_2 = []

for max_depth in range(5, 51, 5):
    hiperparametros2['max_depth'] = max_depth
    modelo = DecisionTreeClassifier(**hiperparametros2)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    resultados_2.append((max_depth, accuracy))

# Paso 7: Mostrar una tabla con el accuracy para los 10 árboles
tabla2 = pd.DataFrame(resultados_2, columns=['max_depth', 'Accuracy (criterion=entropy)'])
print("Tabla 2: Accuracy para los árboles con criterion=entropy")
print(tabla2)
print("\n","#####################################################################","\n")

# Paso 8: Configurar los hiperparámetros del árbol de decisión (criterion=entropy, splitter=random, random_state=123)
hiperparametros3 = {
    'criterion': 'entropy',
    'splitter': 'random',
    'random_state': 123
}

resultados_3 = []

for max_depth in range(5, 51, 5):
    hiperparametros3['max_depth'] = max_depth
    modelo = DecisionTreeClassifier(**hiperparametros3)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    resultados_3.append((max_depth, accuracy))

# Paso 7: Mostrar una tabla con el accuracy para los 10 árboles
tabla3 = pd.DataFrame(resultados_3, columns=['max_depth', 'Accuracy(splitter=random)'])
print("Tabla 3: Accuracy para los árboles con splitter=random")
print(tabla3)
print("\n","#####################################################################","\n")

# Encontrar los hiperparámetros que generan el árbol con mayor precisión (accuracy)
mejor_1 = tabla1.loc[tabla1['Accuracy (criterion=gini)'].idxmax()]
mejor_2 = tabla2.loc[tabla2['Accuracy (criterion=entropy)'].idxmax()]
mejor_3 = tabla3.loc[tabla3['Accuracy(splitter=random)'].idxmax()]

# Mostrar los hiperparámetros que generan el árbol con mayor precisión
print("Hiperparámetros con mayor precisión (criterion=gini):")
print(mejor_1[['max_depth', 'Accuracy (criterion=gini)']])
print()
print("\n","#####################################################################","\n")

print("Hiperparámetros con mayor precisión (criterion=entropy):")
print(mejor_2[['max_depth', 'Accuracy (criterion=entropy)']])
print()
print("\n","#####################################################################","\n")

print("Hiperparámetros con mayor precisión (splitter=random):")
print(mejor_3[['max_depth', 'Accuracy(splitter=random)']])
print()
print("\n","#####################################################################","\n")

# Paso 8: Configurar los hiperparámetros del árbol de decisión (criterion=entropy, splitter=random, random_state=123)
hiperparametros3 = {
    'criterion': 'entropy',
    'splitter': 'random',
    'random_state': 123,
    'min_samples_split': 2  # Cambiar el valor de min_samples_split en el caso 1
}

resultados_3_1 = []

for max_depth in range(5, 51, 5):
    hiperparametros3['max_depth'] = max_depth
    modelo = DecisionTreeClassifier(**hiperparametros3)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    resultados_3_1.append((max_depth, accuracy))

# Mostrar una tabla con el accuracy para los 10 árboles
tabla3_1 = pd.DataFrame(resultados_3_1, columns=['max_depth', 'Accuracy(splitter=random, min_samples_split=2)'])
print("Tabla 3_1: Accuracy para los árboles con splitter=random y min_samples_split=2")
print(tabla3_1)
print()
print("\n","#####################################################################","\n")

# Configurar el valor de min_samples_split en el caso 2
hiperparametros3['min_samples_split'] = 10

resultados_3_2 = []

for max_depth in range(5, 51, 5):
    hiperparametros3['max_depth'] = max_depth
    modelo = DecisionTreeClassifier(**hiperparametros3)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    resultados_3_2.append((max_depth, accuracy))

# Mostrar una tabla con el accuracy para los 10 árboles
tabla3_2 = pd.DataFrame(resultados_3_2, columns=['max_depth', 'Accuracy(splitter=random, min_samples_split=10)'])
print("Tabla 3_2: Accuracy para los árboles con splitter=random y min_samples_split=10")
print(tabla3_2)
print()
print("\n","#####################################################################","\n")

# Obtener el valor máximo de precisión para cada tabla
max_accuracy_3_1 = tabla3_1['Accuracy(splitter=random, min_samples_split=2)'].max()
max_accuracy_3_2 = tabla3_2['Accuracy(splitter=random, min_samples_split=10)'].max()



# Determinar cuál tabla tiene el mejor accuracy
if max_accuracy_3_1 > max_accuracy_3_2:
    mejor_tabla = 'Tabla 3_1: Accuracy con splitter=random y min_samples_split=2'
    mejor_accuracy = max_accuracy_3_1
else:
    mejor_tabla = 'Tabla 3_2: Accuracy con splitter=random y min_samples_split=10'
    mejor_accuracy = max_accuracy_3_2


print("\n","#####################################################################","\n")
# Mostrar el resultado del mejor accuracy
print("El árbol de decisión modificado con el mejor accuracy se encuentra en:")
print(mejor_tabla)
print("El mejor accuracy obtenido es:", mejor_accuracy)

max_accuracy_1 = tabla1['Accuracy (criterion=gini)'].max()
max_accuracy_2 = tabla2['Accuracy (criterion=entropy)'].max()
max_accuracy_3 = tabla3['Accuracy(splitter=random)'].max()

# Determinar cuál tabla tiene el mejor accuracy
if max_accuracy_1 > max_accuracy_2 and max_accuracy_1 > max_accuracy_3:
    mejor_tabla = 'Tabla 1: Accuracy con criterion=gini'
    mejor_accuracy = max_accuracy_1
elif max_accuracy_2 > max_accuracy_1 and max_accuracy_2 > max_accuracy_3:
    mejor_tabla = 'Tabla 2: Accuracy con criterion=entropy'
    mejor_accuracy = max_accuracy_2
else:
    mejor_tabla = 'Tabla 3: Accuracy con splitter=random'
    mejor_accuracy = max_accuracy_3
print("\n","#####################################################################","\n")
# Mostrar el resultado del mejor accuracy
print("El árbol de decisión con el mejor accuracy se encuentra en:")
print(mejor_tabla)
print("El mejor accuracy obtenido es:", mejor_accuracy)
print("\n","#####################################################################","\n")


plt.figure(figsize=(10, 6))
plt.plot(tabla1['max_depth'], tabla1['Accuracy (criterion=gini)'], label='criterion=gini')
plt.plot(tabla2['max_depth'], tabla2['Accuracy (criterion=entropy)'], label='criterion=entropy')
plt.plot(tabla3['max_depth'], tabla3['Accuracy(splitter=random)'], label='splitter=random')

plt.plot(tabla3_1['max_depth'], tabla3_1['Accuracy(splitter=random, min_samples_split=2)'], label='Modificado 1 Samples = 2')
plt.plot(tabla3_2['max_depth'], tabla3_2['Accuracy(splitter=random, min_samples_split=10)'], label='Modificado 2 Samples = 10')


plt.xlabel('Profundidad máxima')
plt.ylabel('Precisión')
plt.title('Precisión para diferentes configuraciones de árboles de decisión')
plt.legend()
plt.grid(True)
plt.show()