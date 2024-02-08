from sklearn.model_selection import GridSearchCV
from group_lasso import GroupLasso
import sklearn.linear_model as lm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

# %% LASSO/ ADAPATIVE LASSO

# Generate Data
p = 20  # Number of parameters
n = 100  # Number of observations
X = np.random.normal(0, 1, (n, p))
n0 = np.random.choice(range(p), 6, replace=False)
beta = np.zeros(p)
beta[n0] = [6.94, -4.03, 1.90, 3.23, 12.26, 9.99]
y = X@beta + np.random.normal(0, 0.5, size=n)
print('Non-zero coefficients: '+', '.join([f'B{i+1}' for i in sorted(n0)]))

# Lasso path
lambdas, lasso_betas, _ = lm.lasso_path(X, y)
lasso_coef = pd.DataFrame(index=lambdas, data=lasso_betas.T)
lasso_coef.columns = [f'B{i}' for i in range(1, p+1)]
non_zero = lasso_coef.abs().mean() > 1e-1
lasso_coef = lasso_coef.loc[:, non_zero]

lassoCV = lm.LassoCV(alphas=lambdas, fit_intercept=False, cv=10)
lassoCV.fit(X, y)
lassoMSEs = pd.Series(lassoCV.mse_path_.mean(1), index=lambdas)
lambda_lasso = lassoMSEs.idxmin()
lasso_coef.plot(logx=True).invert_xaxis()
ymin, ymax = plt.ylim()
plt.vlines(x=lambda_lasso, ymin=ymin, ymax=ymax)
plt.title('Lasso')
plt.show()
lasso_score = np.mean((lassoCV.predict(X)-y)**2)
print(f'Lasso MSE: {lasso_score}, lambda = {lambda_lasso}')
# Adaptive Lasso
gamma = 2

# Adaptive Lasso - OLS
ols_betas = lm.LinearRegression(fit_intercept=False).fit(X, y).coef_
w_ols = ols_betas**-gamma
X_ols = X/w_ols
lambdas, lasso_betas, _ = lm.lasso_path(X_ols, y)
lasso_betas = lasso_betas/w_ols[:, None]
lasso_coef = pd.DataFrame(index=lambdas, data=lasso_betas.T)
lasso_coef.columns = [f'B{i}' for i in range(1, p+1)]
non_zero = lasso_coef.abs().mean() > 1e-1
lasso_coef = lasso_coef.loc[:, non_zero]

lassoCV = lm.LassoCV(alphas=lambdas, fit_intercept=False, cv=10)
lassoCV.fit(X_ols, y)  # As per Zou (2006)
lassoMSEs = pd.Series(lassoCV.mse_path_.mean(1), index=lambdas)
lambda_lasso = lassoMSEs.idxmin()
lasso_coef.plot(logx=True).invert_xaxis()
ymin, ymax = plt.ylim()
plt.title('Adaptive Lasso - OLS')
plt.vlines(x=lambda_lasso, ymin=ymin, ymax=ymax)
plt.show()
lasso_score = np.mean((lassoCV.predict(X_ols)-y)**2)
print(f'Adaptive Lasso - OLS MSE: {lasso_score}, lambda = {lambda_lasso}')

# Adaptive Lasso - Ridge
ridge_betas = lm.RidgeCV(
    fit_intercept=False, alphas=np.logspace(-6, 6, num=100)).fit(X, y).coef_
w_ridge = ridge_betas**-gamma
X_ridge = X/w_ridge
lambdas, lasso_betas, _ = lm.lasso_path(X_ridge, y)
lasso_betas = lasso_betas/w_ridge[:, None]
lasso_coef = pd.DataFrame(index=lambdas, data=lasso_betas.T)
lasso_coef.columns = [f'B{i}' for i in range(1, p+1)]
non_zero = lasso_coef.abs().mean() > 1e-1
lasso_coef = lasso_coef.loc[:, non_zero]

lassoCV = lm.LassoCV(alphas=lambdas, fit_intercept=False, cv=10)
lassoCV.fit(X_ridge, y)  # As per Zou (2006)
lassoMSEs = pd.Series(lassoCV.mse_path_.mean(1), index=lambdas)
lambda_lasso = lassoMSEs.idxmin()
lasso_coef.plot(logx=True).invert_xaxis()
ymin, ymax = plt.ylim()
plt.title('Adaptive Lasso - Ridge')
plt.vlines(x=lambda_lasso, ymin=ymin, ymax=ymax)
plt.show()
lasso_score = np.mean((lassoCV.predict(X_ridge)-y)**2)
print(f'Adaptive Lasso - Ridge MSE: {lasso_score}, lambda = {lambda_lasso}')


# %% GROUP LASSO
# To save effort, variables were dumped to csv file from R

p = spb = 10
x = pd.read_csv('x.csv', index_col=0).values
y = pd.read_csv('y.csv', index_col=0).values
Z = pd.read_csv('Z.csv', index_col=0).values
base_B = pd.read_csv('B.csv', index_col=0).values
group = sum([[i]*10 for i in range(1, 11)], [])
lambdas, _, _ = lm.lasso_path(Z, y)
glreg = GroupLasso(tol=1e-8,groups=group, group_reg=0, l1_reg=0, fit_intercept=True,
        scale_reg='none')
CV = GridSearchCV(glreg, param_grid={
                  'group_reg': lambdas[::2]}, scoring='neg_mean_squared_error', verbose=1)
CV.fit(Z, y)
coef = CV.best_estimator_.coef_.ravel()
lam = CV.best_params_['group_reg']
coef = coef.reshape((p, spb)).T
print(coef)
coef = base_B@coef
coef = pd.DataFrame(coef, index=x.ravel())
non_zero = coef.abs().sum() >5
coef = coef.loc[:,non_zero]
coef.plot()
