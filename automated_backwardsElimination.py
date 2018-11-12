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
