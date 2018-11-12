"""
# Example X, y:

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dat = pd.read_csv('example_data.csv')
X = dat.iloc[:,:-1].values
y = dat.iloc[:,-1].values

# Categorical variables example - Scenario; categories in the 4th column of X:
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
"""

import statsmodels.formula.api as sm

def setVariables(X, y, indicies):
    X_opt = X[:, indicies]
    regressor_ols = sm.OLS(y, X_opt).fit()
    removeIndex = list(regressor_ols.pvalues).index(max(regressor_ols.pvalues))
    return regressor_ols, removeIndex
        
def backwardsElimination(X, y):
    X = np.append(np.ones((X.shape[0], 1)).astype(int), X, 1)
    indicies = list(map(lambda x: x[0], enumerate(X[0])))
    out = 0
    while out == 0:
        regressor_ols, removeIndex = setVariables(X, y, indicies)
        if len(list(filter(lambda x: x > 0.05, list(regressor_ols.pvalues)))) > 0:
            indicies.remove(indicies[removeIndex])
        else:
            out += 1
    return regressor_ols


regressor_be = backwardsElimination(X, y)
