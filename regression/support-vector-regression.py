# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y_shape = y.reshape(10, 1)
y = sc_y.fit_transform(y_shape)

# Fitting the SVR Model to the dataset
# Create your regressor here
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
new_result_value = 6.5
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[new_result_value]]))))

# Visualising the SVR results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Indepedent')
plt.ylabel('Dependent')
plt.show()
