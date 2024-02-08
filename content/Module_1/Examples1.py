# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:42:52 2020

@author: Jonathan Tay (jtay6)
Python implementation of Examples from Module 1 (Functional Data Analysis) of IYSE - 8803 O01 (High Dimensional Data Analytics)
Original source code in R and MATLAB was provided by course instruction staff.
"""

from scipy.ndimage import gaussian_filter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 E261
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from scipy.interpolate import splrep
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

np.random.seed(0)
# %% SPLINES
# * CURVE FITTING
# Data Generation
X = np.linspace(0, 1, 1001)
N = len(X)
Y_true = np.sin(2*np.pi*X**3)**3
Y = Y_true + np.random.normal(0, 0.1, size=N)
# Define knots and basis
k = np.linspace(1/7, 6/7, 6)
H = []
H.append(np.ones((X.shape[0], 1)))
H.append(X.reshape(N, -1))
H.append(X.reshape(N, -1)**2)
H.append(X.reshape(N, -1)**3)
for kk in k:
    H.append(np.maximum((X-kk)**3, 0).reshape(N, -1))
H = np.hstack(H)
# Least square estimates
# "Correct" way
B = np.linalg.lstsq(H, Y)[0]
# Translated from the MATLAB. Avoid forming H.T*H if possible
B2 = np.linalg.solve(H.T@H, H.T@Y)
# Numerically unstable. Not recommended
B3 = np.linalg.inv(H.T@H)@H.T@Y
plt.plot(X, Y, '.', label='Observations')
plt.plot(X, H@B, 'r', label='Estimate')
plt.plot(X, Y_true, 'k', label='True Value')
plt.ylim((-1.5, 1.5))
plt.legend()
plt.grid()
plt.title('Chart on Slide 32')
plt.show()


# %% B-SPLINES
def BSplineBasis(x: np.array, knots: np.array, degree: int) -> np.array:
    '''Return B-Spline basis. Python equivalent to bs in R or the spmak/spval combination in MATLAB.
    This function acts like the R command bs(x,knots=knots,degree=degree, intercept=False)
    Arguments:
        x: Points to evaluate spline on, sorted increasing
        knots: Spline knots, sorted increasing
        degree: Spline degree. 
    Returns:
        B: Array of shape (x.shape[0], len(knots)+degree+1). 
    Note that a spline has len(knots)+degree coefficients. However, because the intercept is missing 
    you will need to remove the last 2 columns. It's being kept this way to retain compatibility with
    both the matlab spmak way and how R's bs works.

    If K = length(knots) (includes boundary knots)
    Mapping this to R's bs: (Props to Nate Bartlett )
    bs(x,knots,degree,intercept=T)[,2:K+degree] is same as BSplineBasis(x,knots,degree)[:,:-2]
    BF = bs(x,knots,degree,intercept=F) drops the first column so BF[,1:K+degree] == BSplineBasis(x,knots,degree)[:,:-2]
    '''
    nKnots = knots.shape[0]
    lo = min(x[0], knots[0])
    hi = max(x[-1], knots[-1])
    augmented_knots = np.append(
        np.append([lo]*degree, knots), [hi]*degree)
    DOF = nKnots + degree +1 # DOF = K+M, M = degree+1
    spline = BSpline(augmented_knots, np.eye(DOF),
                     degree, extrapolate=False)
    B = spline(x)
    return B
#raise
'''
# * B-SPLINE BASIS
# Basis
n = 100
nKnots = 10
x = np.arange(1, n+1)
knots = np.linspace(1, n, nKnots)
Bs = []  # Spline bases for each order
for degree in range(4):  # 0th order to cubic
    Bs.append(BSplineBasis(x, knots, degree)[:,:-2])
    # raise
    plt.figure()
    plt.plot(Bs[-1])
    plt.ylim((0, 1.))
    plt.title(f'B-Spline Basis, degree = {degree}')
    plt.show()

# Example code plots the coefficients of the cubic spline basis
plt.matshow(Bs[-1], aspect=(nKnots+degree)/100)  # type: ignore
plt.title('B-Matrix')
plt.xlabel('Spline index')  # index of spline
plt.ylabel('x')  # x-coordinate
# * CURVE FITTING
# Generate data:
n = 100
x = np.linspace(0, 1, n)
y = 2.5*x-np.sin(10*x)-np.exp(-10*x)
sigma = 0.3
ynoise = y + np.random.normal(0, sigma, size=n)
# Generate B-spline basis:
B = BSplineBasis(x, np.linspace(0, 1, 10), 2)[:,:-2]
# Least square estimation
BBBB = B@np.linalg.inv(B.T@B)@B.T
yhat = B@np.linalg.pinv(B)@ynoise  # More numerically stable
yhat2 = np.linalg.lstsq(B.T, B.T@ynoise)[0]  # Probably the recommended way
yhat3 = BBBB@ynoise  # naive way
assert np.isclose(yhat, yhat2).all()
assert np.isclose(yhat, yhat3).all()
K = np.trace(BBBB)
sigma2 = (1/(n-K))*(ynoise-yhat).T@(ynoise-yhat)
width = np.diag(sigma2*BBBB)**0.5
yn = yhat-3*width
yp = yhat+3*width
plt.figure()
plt.plot(x, ynoise, 'r.', label='Observations')
plt.plot(x, yn, 'b--', label='Confidence Band')
plt.plot(x, yp, 'b--')
plt.plot(x, yhat, 'k', label='Estimate')
plt.legend()
plt.title('Smoother Example')
plt.show()

# * FAT CONTENT PREDICTION
# Baseline linear model
nTest = 20
meat = pd.read_csv("meat.csv", index_col=0).values
# If no pandas: meat = np.loadtxt('meat.csv',delimiter=',',skiprows=1,usecols=np.arange(1,102))
np.random.shuffle(meat)
train = meat[:-nTest, :]
test = meat[-nTest:, :]

regressor = LinearRegression()
y = train[:, -1]
X = train[:, :-1]
regressor.fit(X, y)
pred1 = regressor.predict(test[:, :-1])
mse1 = sum(((test[:, -1]-pred1)**2))/nTest
print(f'Fat content, MSE using raw features: {mse1}')

# B-splines
X = meat[:, :-1]
deg = 3
dim = 10
xx = np.linspace(0, 1, 100)
knots = np.linspace(0, 1, 8)
B = BSplineBasis(xx, knots, deg)[:,:-2]
Bcoef = np.linalg.lstsq(B, X.T)[0].T
meat2 = np.hstack([Bcoef, meat[:, -1][:, None]])

train = meat2[:-nTest, :]
test = meat2[-nTest:, :]

regressor = LinearRegression()
y = train[:, -1]
X = train[:, :-1]
regressor.fit(X, y)
pred2 = regressor.predict(test[:, :-1])
mse2 = sum(((test[:, -1]-pred2)**2))/nTest
print(f'Fat content, MSE using spline features: {mse2}')

# Equivalent B-spline model using library function:
X_transformed = []
for signal in meat[:, :-1]:
    # use internal knots only
    t, c, k = splrep(xx, signal, task=-1, t=knots[1:-1], k=3)
    X_transformed.append(c[:dim].reshape(-1, dim))  # remove 0s arising from knot padding
X_transformed = np.vstack(X_transformed)
y = meat[:, -1]
meat2 = np.hstack([X_transformed, y.reshape(-1, 1)])
train = meat2[:-nTest, :]
test = meat2[-nTest:, :]

regressor = LinearRegression()
y = train[:, -1]
X = train[:, :-1]
regressor.fit(X, y)
pred3 = regressor.predict(test[:, :-1])
mse3 = sum(((test[:, -1]-pred3)**2))/nTest

assert np.isclose(pred2, pred3).all()  # Equivalent

# %% SMOOTHING SPLINES
# * OVER-FITTING
# Generate data:
n = 100
D = np.linspace(0, 1, n)
k = 40
sigma = 0.3
y = 2.5 * D - np.sin(10 * D) - np.exp(-10 * D) + np.random.normal(size=n)*sigma
# Generate B-spline basis:
knots = np.linspace(0, 1, k)
B = BSplineBasis(D, knots, 2)[:,:-2]
# Least Square Estimation:
yhat = B@np.linalg.lstsq(B, y)[0]
assert np.isclose(B@np.linalg.inv(B.T@B)@B.T@y, yhat).all()
plt.plot(D, y, 'r*', label='Noisy data')
plt.plot(D, yhat, 'k-', label='Quadratic Spline - no smoothing')
plt.legend()
plt.show()
# Smoothing Penalty
# Different lambda selection
B2 = np.diff(B, axis=0, n=2)*(n-1)**2  # Numerical derivative
omega = B2.T.dot(B2)/(n-2)
lams = np.arange(0, 1e-3, 1e-6)
p = len(lams)
RSS = []
df = []
for lam in lams:
    S = B@np.linalg.inv(B.T@B+lam*omega)@B.T  # Not great but we still need to get the trace of S
    yhat = S.dot(y)
    RSS.append(((yhat-y)**2).sum())
    df.append(np.trace(S))
RSS = np.array(RSS)
df = np.array(df)
# GCV criterion
GCV = (RSS/n)/(1-df/n)**2
i = np.argmin(GCV)
m = GCV[i]
plt.plot(lams, GCV, label='GCV')
plt.plot(lams[i], m, 'r*', label='Lowest GCV')
plt.title("Generalised Cross Validation")
plt.xlabel('$\lambda$')  # noqa: W605
plt.ylabel('GCV Score')
plt.legend()
plt.show()
S = B@np.linalg.inv(B.T@B+lams[i]*omega)@B.T
yhat = S@y
plt.plot(D, y, 'r*', label='Noisy data')
plt.plot(D, yhat, 'k-', label='Spline - best smoother')
plt.legend()
plt.title("Smoothing Splines")
plt.show()

# %% KERNEL SMOOTHERS
# * RBF KERNEL
x = np.arange(101)
y = (np.sin(x/10)+(x/50)**2+0.1*np.random.normal(0, 1, 101))


def kerf(z):
    return np.exp(-z*z/2)*(2*np.pi)**-0.5


# leave-one-out CV
bandwidths = np.arange(1, 4.001, 0.1)
MSEs = []
for w in bandwidths:
    loo = LeaveOneOut()
    errs = []
    for trg, tst in loo.split(x):
        z = kerf((x[tst]-x[trg])/w)
        yke = np.average(y[trg], weights=z)
        errs.append(y[tst]-yke)
    MSEs.append(sum([err**2 for err in errs]))
MSEs = np.array(MSEs).squeeze()
w_star = bandwidths[np.argmin(MSEs)]
plt.plot(bandwidths, MSEs, label='Validation Curve')
plt.plot(w_star, min(MSEs),'r+', label='Lowest LOO MSE')
plt.title('Leave-One-Out Validation Curve, Kernel Estimator')
plt.xlabel('Kernel Bandwidth')
plt.ylabel('MSE')
plt.legend()
plt.show()

w_star = bandwidths[np.argmin(MSEs)]

# Interpolation for N values
N = 1000
xx = np.linspace(min(x), max(x), N)
yy = []
for xx_ in xx:
    z = kerf((xx_-x)/w_star)
    yy.append(np.average(y, weights=z))

ytrue = np.sin(xx/10)+(xx/50)**2
plt.plot(x, y, 'r.', label='Noisy observations')
plt.plot(xx, ytrue, 'b', label='True mean')
plt.plot(xx, yy, 'k', label='  RBF Kernel Regression')
plt.legend()
plt.title('RBF Kernel Smoother')
plt.show()

'''
# %% FUNCTIONAL PRINCIPAL COMPONENTS
# *FUNCTIONAL DATA CLASSIFICATION
# Data generation
M = 100
n = 50
x = np.linspace(0, 1, n)
E = distance_matrix(x[:, None], x[:, None])
Sigma = np.exp(-10*E**2)
eig_vals, eig_vecs = np.linalg.eigh(Sigma)
Sigma_sqrt = eig_vecs@np.diag((eig_vals+1e-8)**0.5)@eig_vecs.T
S_noise = np.exp(-0.1*E**2)
noise_eval, noise_evec = np.linalg.eigh(S_noise)
S_sqrt_noise = noise_evec@np.diag((noise_eval+1e-8)**0.5)@noise_evec.T
# Class 1
mean1 = Sigma_sqrt@np.random.normal(size=n)
noise1 = S_sqrt_noise@np.random.normal(size=n)
signal1 = mean1 + noise1
var1 = np.var(signal1)
ds1 = (var1/100)**0.5
S_sqrt_err1 = np.eye(n)*ds1
x1 = []
for i in range(M):
    noise1 = S_sqrt_noise@np.random.normal(size=n)
    error1 = S_sqrt_err1@np.random.normal(size=n)
    x1.append(mean1+noise1+error1)
x1 = np.vstack(x1)
plt.plot(x, x1.T, 'b')
plt.title('Class 1')
plt.show()
# Class 2
mean2 = Sigma_sqrt.dot(np.random.normal(size=n))
noise2 = S_sqrt_noise.dot(np.random.normal(size=n))
signal2 = mean2 + noise2
var2 = np.var(signal2)
ds2 = (var2/100)**0.5
S_sqrt_err2 = np.eye(n)*ds2
x2 = []
for i in range(M):
    noise2 = S_sqrt_noise@np.random.normal(size=n)
    error2 = S_sqrt_err2@np.random.normal(size=n)
    x2.append(mean2+noise2+error2)
x2 = np.vstack(x2)
plt.plot(x, x2.T, 'r')
plt.title('Class 2')
plt.show()
# Train and test data sets
X = np.vstack([x1, x2])
lab = np.ones(2*M)
lab[:M] = 0
indices = np.arange(2*M)
X_trg, X_tst, Y_trg, Y_tst, i_trg, i_tst = train_test_split(
    X, lab, indices, train_size=0.8)
# Option 1: B-splines
knots = np.linspace(0, 1, 8)
B = BSplineBasis(x, knots, degree=3)[:,:-2]
print(B.shape, X.T.shape)
Bcoef = np.linalg.lstsq(B, X.T)[0].T
print(Bcoef.shape)
rf = RandomForestClassifier()
rf.fit(Bcoef[i_trg, :], Y_trg)
pred = rf.predict(Bcoef[i_tst, :])
conf = confusion_matrix(Y_tst, pred)
conf = pd.DataFrame(conf, index=['Class 0', 'Class 1'], columns=[
                    'Predicted 0', 'Predicted 1'])
print('Confusion Matrix - Splines\n', conf)
# Option 2: Functional principal components
B_stacked = np.tile(B.T, 200).T
X_stacked = X.ravel()
print(B_stacked.shape, X_stacked.shape)
beta = np.linalg.lstsq(B_stacked, X_stacked)[0]
mu_hat = B.dot(beta)  # Mean function via B-Spline
mu_hat2 = gaussian_filter(X.mean(0),3)
print("mu shape:", mu_hat2.shape)
print("X:", X.shape)
plt.plot(x, X.mean(0), 'k+-', label='Mean of data')
plt.plot(x, mu_hat, 'b', label='Mean function via Spline')
plt.plot(x, mu_hat2, 'm', label='Mean function via Smoothing data mean')
plt.plot(x, mean1, 'g', label='Class 0')
plt.plot(x, mean2, 'r', label='Class 1')
plt.legend()
plt.title("Mean Estimate")
plt.show()
diffs = X-mu_hat
Cov = np.cov(diffs.T)
grids = np.meshgrid(x, x)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(grids[0], grids[1], Cov, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.title('Unsmoothed Covariances')
plt.show()
# additional Cov smoothing:
Cov = gaussian_filter(Cov, sigma=7)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(grids[0], grids[1], Cov, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.title('Smoothed Covariances')
plt.show()
l, psi = np.linalg.eigh(Cov)
print(psi.shape)
PCs = psi[:, -2:]
FPC_scores = diffs.dot(PCs)
plt.plot(FPC_scores[:M,0],FPC_scores[:M,1],'bo',label='Class 1')
plt.plot(FPC_scores[M:,0],FPC_scores[M:,1],'ro',label='Class 2')
plt.legend()
plt.show()
rf = RandomForestClassifier()
rf.fit(FPC_scores[i_trg, :], Y_trg)
pred = rf.predict(FPC_scores[i_tst, :])
conf = confusion_matrix(Y_tst, pred)
conf = pd.DataFrame(conf, index=['Class 0', 'Class 1'], columns=[
                    'Predicted 0', 'Predicted 1'])
print('Confusion Matrix - FPCA\n', conf)
