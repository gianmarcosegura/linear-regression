# -*- coding: utf-8 -*-

# Regresión linear simple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')

# Dividir
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

# Conjunto entrenamiento - test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Escalado de variables

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Crear modelo de regresión simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test

y_pred = regression.predict(X_test)


# Visualizar los resultados de entrenamiento

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo - Años de experiencia (Entrenamiento)')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (en €)')
plt.show()



# Visualizar los resultados de test

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo - Años de experiencia (Testing)')
plt.xlabel('Años de experiencia')
plt.ylabel('Sueldo (en €)')
plt.show()