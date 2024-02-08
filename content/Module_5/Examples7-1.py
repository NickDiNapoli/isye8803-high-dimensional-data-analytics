from scipy.optimize import fminbound
from scipy.linalg import sqrtm
from sklearn.linear_model import OrthogonalMatchingPursuit
import pywt
import numpy as np
from scipy.signal import gausspulse
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.linalg import dft
import cvxpy as cp
from scipy.linalg import orth
from time import clock
from scipy.io import loadmat
from scipy.interpolate import BSpline
from skimage.filters import threshold_otsu


def l1eq_pd(x0, A, b, verbose=False):
    '''Solves min_x ||x||_1 st Ax = b via CVXPY
    
    '''
    n = len(x0)
    x = cp.Variable(n, complex=True)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [A@x == b]
    prob = cp.Problem(objective, constraints)
    # SCS is waay faster than default ECOS. Dunno why.
    result = prob.solve(solver='SCS', verbose=verbose)
    return x.value


# %% Example 1 - Sparsity in Time domain

# Generate signal
T = np.arange(-15e-9, 15e-9+np.finfo(float).eps, 1/15e9)
x = gausspulse(T, fc=4e9, bw=0.5)
T += 15e-9
plt.figure()
plt.plot(T, x)
plt.title('Time Domain')
plt.show()
# FFT
xf = fft(x)
plt.figure()
plt.plot(np.abs(xf))
plt.title('Frequency domain')
# Compressive Sensing
N = len(T)
K = 90  # Samples
B = dft(N)  # Fourier matrix
q = np.random.choice(N, K, replace=False)  # random rows
A = B[q]  # Select rows
y = A@x  # Measure frequencies (at random)
x0 = A.T@y  # initial guess
xp = l1eq_pd(x0, A, y)  # Recover
plt.figure()
plt.plot(T, xp)
plt.title('Recovered Signal')


# %% Example 2: Sparsity in frequency domain
# Generate signal
N = 1024
n = np.arange(N)
k1 = 30
k2 = 80
k3 = 100
x = (np.sin(2*np.pi*(k1/N)*n)+np.sin(2*np.pi*(k2/N)*n)+np.sin(2*np.pi*(k3/N)*n))
plt.figure()
plt.plot(x)
plt.title('Time Domain')
# FFT analysis
xf = fft(x)
plt.figure()
plt.plot(np.abs(xf))
plt.title('Frequency domain')
# Compressive sensing
K = 650  # Samples
ID = np.eye(N)
q = np.random.choice(N, K, replace=False)  # random rows
Phi = ID[q]
Psi = dft(N)
xf = Psi@x
y = Phi@x  # taking random time measurements
x0 = Psi.T@Phi.T@y  # Calculating Initial guess1
xp = l1eq_pd(x0, Phi@Psi, y, verbose=False)  # Running the recovery Algorithm
xprec = np.real(-np.linalg.inv(Psi)@xp)  # recovered signal in time domain
plt.figure()
plt.plot(xprec)
plt.title('Recovered Signal')

# %% Example 3 - CS Applications in Images (Wavelets)
X = plt.imread('lena256.bmp')*1.
a, b = X.shape
M = 190
R = np.random.normal(size=(M, a))
Y = R@X

N = a


def dwt(N):
    g, h = pywt.Wavelet('sym8').filter_bank[:2]
    L = len(h)  # Length of bandwidth
    rank_max = int(np.log2(N))  # Maximum Layer
    rank_min = int(np.log2(L))+1  # Minimum Layes
    ww = np.eye(2**rank_max)  # Proprocessing Matrix

    for jj in range(rank_min, rank_max+1):
        nn = 2**jj
        # Construct vector
        p1_0 = np.concatenate([g, np.zeros(nn-L)])
        p2_0 = np.concatenate([h, np.zeros(nn-L)])
        p1 = []
        p2 = []
        # Circular move
        for ii in range(2**(jj-1)):
            shift = 2*ii
            p1.append(np.roll(p1_0, shift))
            p2.append(np.roll(p2_0, shift))
        p1 = np.stack(p1)
        p2 = np.stack(p2)
        # Orthogonal Matrix
        w1 = np.concatenate([p1, p2])
        wL = len(w1)
        w = np.eye(2**rank_max)
        w[:wL, :wL] = w1
        ww = ww@w
    return ww


ww = dwt(N)

#  Measure value
Y = Y@ww.T
# Measure Matrix
R = R@ww.T

reg = OrthogonalMatchingPursuit(n_nonzero_coefs=256,tol=1e-5, fit_intercept=False, normalize=False)
X2 = np.zeros((a,b))
for i in range(b):
    reg.fit(R, Y[:, i])
    X2[:, i] = reg.coef_
# original Image
plt.figure()
plt.imshow(X, cmap='gray')
plt.title('original Image')
plt.show()

# Transfered Image
plt.figure()
plt.imshow(np.clip(X2, 0, 255).astype('uint8'), cmap='gray')
plt.title('Transferred Image')
plt.show()


# Recovered image
plt.figure()
X3 = ww.T@X2@ww  # inverse DWT

plt.imshow(np.clip(X3, 0, 255).astype('uint8'), cmap='gray')
plt.title('Recovered Image')
plt.show()


# %% Example 4 - Noisy Signal Recovery

def l1dantzig_pd(A, y, eps, verbose=False):
    '''Solves min_x ||x||_1 st --eps <= A.T@Ax - b <= eps via CVXPY
    Replaces l1dantzig_pd.m
 '''
    assert eps >= 0
    m, n = A.shape
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [-A.T@(A@x - y)-eps <= 0, A.T@(A@x - y)-eps <= 0]
    prob = cp.Problem(objective, constraints)
    # When we have no risk of complex numbers, ECOS > SCS
    result = prob.solve(solver='ECOS', verbose=verbose)
    return x.value


def l1dantzig_pdLP(A, y, eps, verbose=True):
    '''Solves min_x ||x||_1 st --eps <= A.T@Ax - b <= eps via CVXPY
    Replaces l1dantzig_pd.m
    '''
    assert eps >= 0
    m, n = A.shape
    x = cp.Variable(n)
    u = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.sum(u))
    constraints = [-A.T@(A@x - y)-eps <= 0,
                   A.T@(A@x - y)-eps <= 0,
                   x <= u,
                   -x-u <= 0]
    prob = cp.Problem(objective, constraints)
    # When we have no risk of complex numbers, ECOS > SCS
    result = prob.solve(solver='GLPK', verbose=verbose)
    return x.value


# signal length
N = 512

# number of spikes to put down
T = 20

#  number of observations to make
K = 120

#  random +/- 1 signal
x = np.zeros(N)
ind = np.random.choice(N, size=T, replace=False)
x[ind] = np.sign(np.random.rand(T)-0.5)


#  measurement matrix = random projection
print('Creating measurment matrix...')
A = np.random.randn(K, N)
A = orth(A.T).T
print('Done.')

# % noisy observations
sigma = 0.005
e = sigma*np.random.randn(K)
y = A@x + e

plt.plot(x)
plt.title('original')
plt.show()

plt.imshow(A)
plt.title('Measurement Matrix')
plt.show()

#  Dantzig selection
epsilon = 3e-3
st = clock()
xp = l1dantzig_pdLP(A, y, epsilon)
print(clock()-st)
plt.plot(xp-x)
plt.title('Recovered vs Original (Errors)')
plt.show()


# %% Matrix Completion

# Generating original matrix
n1 = 10
n2 = 8
A = np.random.randint(-20, 21, size=(n1, n2))
r = 2
u, s, vh = np.linalg.svd(A)
s[r:] = 0
X = (u[:, :n2]*s)@vh  # See doc for svd for instructions to rebuild A using u,s,vh
X0 = X.copy()  # X, X0 are rank r matrices of random (almost) ints

# Removing 20% of the observations
mask = np.random.rand(n1, n2) >= 0.8
X[mask] = 0
m = (~mask).sum()
# %Initialization
Y = np.zeros((n1, n2))
delta = n1*n2/m
tau = 250
# %Iterations
vec = []
err = []
for i in range(500):
    u, s, vh = np.linalg.svd(Y)
    s_t = np.maximum(s-tau, 0)
    Z = (u[:, :n2]*s_t)@vh
    P = X-Z
    P[mask] = 0
    Y0 = Y.copy()
    Y = Y0 + delta*P
    # raise
    vec.append(((Y-Y0)**2).sum())
    err.append(((X0-Z)**2).sum()/(X0**2).sum())

# % plot the results
plt.figure()
plt.plot(vec)
plt.title('Change in Y ')
plt.figure()
plt.plot(err)
plt.title('Error vs. true signal')
plt.figure()
Xr = X0[mask]
Zr = Z[mask]
plt.plot(Xr, label='Signal')
plt.plot(Zr, 'r', label='Recovered')
plt.title('Missing Observations')
plt.legend()
plt.figure()
plt.plot(Zr-Xr)
plt.title('Error of missing observations')
plt.figure()
plt.imshow(Z)
plt.title('Recovered Matrix')
plt.figure()
plt.imshow(X0)
plt.title('Original Matrix')

# %% Robust PCA
D = plt.imread('building.PNG')@np.array([0.2989, 0.5870, 0.1140, 0.])
D = np.round(255*D)
plt.imshow(D, vmin=0, vmax=255, cmap='gray')
lam = 1e-2
m, n = D.shape
tol = 1e-7
maxIter = 1000

# % Initialize A,E,Y,u
Y = D.copy()
norm2 = np.linalg.norm(Y, 2)
normInf = Y.max()/lam
# Y is Lagrangian multiplier for each pixel constaint that the
# reconstructed values must match truth
Y = Y/normInf

A_hat = np.zeros((m, n))  # Smooth component, L "low rank"
E_hat = np.zeros_like(A_hat)  # Noise component, S "sparse"
# Tune these
mu = 1.25/norm2
mu_bar = mu*1e7
rho = 1.5
d_norm = np.linalg.norm(D, 'fro')
iter = 0
total_svd = 0
converged = False
stopCriterion = 1
# D is M the original data.
while not converged:
    iter += 1
    X = D - A_hat + Y/mu
    # temp_T = D-A_hat+ Y/mu
    # Soft threshold on scalars (this implementation matches slides better)
    E_hat = np.sign(X)*np.maximum(np.abs(X)-lam/mu, 0)
    # Equivalent, as per MATLAB code
    # E_hat = np.maximum(X-lam/mu,0)
    # E_hat = E_hat+np.minimum(X+lam/mu,0)
    # Soft threshold SVD
    u, s, vh = np.linalg.svd(D-E_hat+Y/mu,full_matrices=False)
    svp = (s > 1/mu).sum()
    A_hat = (u[:, :svp]*(s[:svp]-1/mu))@vh[:svp]  # reconstruct
    total_svd += 1
    Z = D-A_hat-E_hat  # Constraint violation
    Y = Y + mu*Z  # update lagragian multipliers
    mu = min(mu*rho, mu_bar)  # mu is augmented penalty coefficient
    # % stop Criterion
    stopCriterion = np.linalg.norm(Z, 'fro') / d_norm
    if stopCriterion < tol:
        converged = True

plt.figure()
plt.imshow(A_hat.astype('uint8'), cmap='gray')
plt.title("Smooth Component (Low Rank)")
plt.figure()
E_hat = np.clip(E_hat, 0, 255)
plt.imshow(E_hat.astype('uint8'), cmap='gray')
plt.title("Noise Component (Sparse)")

# %% Smooth Sparse Decomposition


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
    DOF = nKnots + degree + 1  # DOF = K+M, M = degree+1
    spline = BSpline(augmented_knots, np.eye(DOF),
                     degree, extrapolate=False)
    B = spline(x)
    return B


def thresh(x, t, tau):
    assert t in ['s', 'h']

    if t is 't':
        tmp = x.copy()
        tmp[np.abs(tmp) < tau] = 0
        return tmp
    else:
        return np.sign(x)*np.maximum(np.abs(x)-tau, 0)


d = loadmat('data.mat')
A0 = d['A0']
Y0 = d['Y0']
sigma = 0.05
delta = 0.2
Y = Y0 + delta*A0 + sigma*np.random.randn(*Y0.shape)
plt.figure()
plt.imshow(Y,cmap='jet')
plt.title('Corrupted Data')
kx = 6
ky = 6
nx, ny = Y.shape
B1 = BSplineBasis(np.arange(nx), np.linspace(0, nx-1, kx), 2)[:, :-2]
B2 = BSplineBasis(np.arange(ny), np.linspace(0, ny-1, ky), 2)[:, :-2]
snk = 4
skx = int(np.round(nx/snk))
sky = int(np.round(ny/snk))
Bs1 = BSplineBasis(np.arange(nx), np.linspace(0, nx-1, skx), 1)[:, :-2]
Bs2 = BSplineBasis(np.arange(ny), np.linspace(0, ny-1, sky), 1)[:, :-2]


def splinegcv(lam, Y, C, Z, nmiss, W):
    # % Estimate Generalized Cross-validation value

    ndim = len(np.squeeze(Y).shape)
    H = []
    dfi = np.zeros(ndim)
    for idim in range(ndim):
        # print(ndim,idim)
        L1 = C[idim].shape[0]
        # o = np.ones(L1)+lam*np.diag(C[idim])
        o = 1+lam*np.diag(C[idim])
        tmp = Z[idim]@np.diag(1/o)@Z[idim].T
        H.append(tmp)

        dfi[idim] = sum(1/(1+lam*np.diag(C[idim])))

    df = np.product(dfi)
    if ndim == 1:
        Yhat = H[0]@Y
    elif ndim == 2:
        # print(H[0].shape,H[1].shape,Y.shape)
        Yhat = H[0]@Y@H[1]
    elif ndim >= 3:
        raise NotImplementedError
        # Yhat = double(ttm(tensor(Y),H));

    if not W:
        RSS = ((Y-Yhat)**2).sum()
    else:
        diff = Y-Yhat
        RSS = (diff*W*diff).sum()

    n = len(Y)
    GCVscore = RSS/(n-nmiss)/(1-df/n)**2
    return GCVscore


y = Y
B = [B1, B2]
Ba = [Bs1, Bs2]
lam = []
gamma = []
maxIter = 20
errtol = 1e-6


def bsplineSmoothDecompauto(y, B, Ba, lam, gamma, maxIter=20, errtol=1e-6):
    def plus0(x): return np.maximum(x, 0)
    def norm(x): return np.linalg.norm(x, 2)
    sizey = y.shape
    ndim = len(y.squeeze().shape)

    if ndim == 1:
        Lbs = 2*norm(Ba[0])**2
        X = np.zeros(Ba[0].shape[1])
        a = 1
        BetaA = X.copy()
    elif ndim == 2:
        Lbs = 2*norm(Ba[0])**2*norm(Ba[1])**2
        X = np.zeros((Ba[0].shape[1], Ba[1].shape[1]))
        BetaA = X.copy()

    if len(lam) == 1:
        lam = np.ones(ndim)*laml

    SChange = 1e10
    H = []
    a = np.zeros_like(y)
    C = []
    Z = []

    for idim in range(ndim):
        Li = sqrtm(B[idim].T@B[idim])
        Li = Li + 1e-8*np.eye(*Li.shape)
        Di = np.diff(np.eye(B[idim].shape[1]), 1, axis=0)
        tmp = np.linalg.pinv(Li.T)@(Di.T@Di)@np.linalg.pinv(Li)
        Ui, ctmp, _ = np.linalg.svd(tmp)
        C.append(np.diag(ctmp))
        Z.append(B[idim]@np.linalg.pinv(Li.T)@Ui)

    iIter = 0
    t = 1

    while SChange > errtol and iIter < maxIter:
        iIter += 1
        Sold = a
        BetaSold = BetaA
        told = t
        def gcv(x): return splinegcv(x, y, C, Z, 0, [])

        if len(lam) == 0 and iIter == 1:
            lam = fminbound(gcv, 1e-2, 1e3)
            lam = lam*np.ones(ndim)

        # % %
        H = []
        for idim in range(ndim):
            L1 = C[idim].shape[0]
            o = np.ones(L1)+lam[idim]*np.diag(C[idim])
            tmp = Z[idim]@np.diag(1/o)@Z[idim].T
            H.append(tmp)
        if ndim == 1:
            yhat = H[0]@(y-a)
            BetaSe = X + 2/Lbs*Ba[0].T@(y - Ba[0]@X - yhat)
        elif ndim == 2:
            yhat = H[0]@(y-a)@H[1]
            BetaSe = X + 2/Lbs*Ba[0].T@(y - Ba[0]@X@Ba[1].T - yhat)@Ba[1]

        maxYe = np.abs(BetaSe).max()

        # %
        if not gamma and iIter % 3 == 1:
            gamma = threshold_otsu(np.abs(BetaSe)/maxYe)*maxYe*Lbs

        # change 'h' to 's' for softthresholding
        BetaA = thresh(BetaSe, 'h', gamma/Lbs)
        if ndim == 1:
            a = Ba[0] @BetaA
        elif ndim == 2:
            a = Ba[0] @BetaA@ Ba[1].T
        t = (1+(1+4*told**2)**0.5)/2

        if iIter == 1:
            X = BetaA
        else:
            X = BetaA+(told-1)/t*(BetaA-BetaSold)

        SChange = a-Sold
        SChange = (SChange**2).sum()

    return yhat, a


yhat, a = bsplineSmoothDecompauto(Y, B, Ba, [], [])

plt.figure()
plt.imshow(yhat,cmap='jet')
plt.title('Smooth (Mean)')
plt.show()

plt.figure()
plt.imshow(a,cmap='jet')
plt.title('Sparse (Anomalies)')
plt.show()


# %% Kernel Ridge Regression %%%%%%%%%%%%%%%%%%
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist
Xtrain = np.arange(0.01,1.001,0.01)
Yt = np.sin(Xtrain*10)+(Xtrain*2)**2
Ytrain = Yt + 0.2*np.random.randn(100)
N = 100
Xtest = np.linspace(min(Xtrain),max(Xtrain),N);
n = len(Xtrain)
lam = 0.04
c = 0.04
kernel1 = np.exp(-cdist(Xtrain[:,None],Xtrain[:,None])**2 / (2*c)) #train kernel
kernel2 = np.exp(-cdist(Xtrain[:,None],Xtest[:,None])**2 / (2*c)) # test kernel
reg = Ridge(alpha = lam)
reg.fit(kernel1,Ytrain)
yhatRBF = reg.predict(kernel2)
plt.plot(Xtrain,Yt,'g-',label='True Function')
plt.plot(Xtrain,Ytrain,'bo',label='Noisy Obsverations')
plt.plot(Xtrain,yhatRBF,'r--',label='RBF Ridge Estimate')

