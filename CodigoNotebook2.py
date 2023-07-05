import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
import warnings



#Ignorar las advertencias
warnings.filterwarnings("ignore")

# Paso 1: Leer el archivo stroke.csv
datos = pd.read_csv('stroke.csv')

# Paso 2: Seleccionar aleatoriamente el 80% del conjunto de datos para entrenar y el 20% restante para las pruebas
X_train, X_test, y_train, y_test = train_test_split(datos.drop('stroke', axis=1), datos['stroke'], test_size=0.2, random_state=123)

# Paso 3: llenar los datos faltantes y normalizar los datos
categoricas = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numericas = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
