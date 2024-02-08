import matplotlib.pyplot as plt
import tensorly as tl
from itertools import product
import numpy as np
import tensorly.decomposition

# %% TENSOR PRELIMNARIES I
#  Inner product

X = np.arange(1, 25).reshape(4, 2, 3)
Y = np.arange(25, 49).reshape(4, 2, 3)
Z = tl.tenalg.inner(X, Y)

#  Matricization
X1 = np.matrix([[1, 3], [2, 4]])
X2 = np.matrix([[5, 7], [6, 8]])
X = np.ones([2,2,2])
X[0,:,:] = X1
X[1,:,:] = X2
A = tl.unfold(X, 1)  
B = tl.unfold(X, 2)  
C = tl.unfold(X, 0)  


# %% TENSOR PRELIMNARIES II

# n-mode vector product
ML_mode = 1
A = np.ones(2)
Y = tl.tenalg.mode_dot(X, A, 3-ML_mode).T

#  n-mode matrix product
A = np.eye(2)
Y = tl.tenalg.mode_dot(X, A, 3-ML_mode).T

#  Kronecker product
A = np.arange(1, 7).reshape(3, -1)
B = np.arange(1, 7).reshape(2, -1)
C = tl.kron(A, B)


# Khatri-Rao Poduct
A = np.arange(1, 7).reshape(3, -1)
B = np.arange(1, 5).reshape((2, 2))
C = tl.kr([A, B])

#  Hadamard Product
A = np.arange(1, 7).reshape(3, -1)
B = A.copy()
C = A*B


# %% TENSOR DECOMPOSITION I AND II (Combined)
import tensorly.decomposition

#  CP decomposition
X = np.random.uniform(size=(5, 4, 3))
P = tl.decomposition.parafac(X, 2, verbose=False)
X_estimated = tl.kruskal_to_tensor(P)  # Reconstruct
err = ((X-X_estimated)**2).mean()
print(f'CP-Error = {err}')


'''#  Tucker decomposition 
tucker = tl.decomposition.tucker(X,[2,2,1])  
X_estimated = tl.tucker_to_tensor(*tucker) # Reconstruct
err = ((X-X_estimated)**2).mean()
print(f'Tucker-Error = {err}')'''

#  Heat transfer example
# Data Generation
L = 0.05
H = 0.05
dx = 0.0025
dy = 0.0025
tmax = 10
dt = 0.01
epsilon = 0.0001
alpha = 0.5e-5+np.random.random()*1e-5
SimulateData = []
SimulateDataNoNoise = []
r_x = alpha*dt/dx**2
r_y = alpha*dt/dy**2
fo = r_x + r_y
if fo > 0.5:
    msg = f'Current Fo = {fo}, which is numerically unstable (>0.5)'
    raise ValueError(msg)
# x, y meshgrid based on dx, dy
nx = int(L/dx + 1)
ny = int(H/dy + 1)
X, Y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, H, ny))
# center point of the domain
ic = int((nx-1)/2)
jc = int((ny-1)/2)

# initial and boundary conditions
S = np.zeros((ny, nx))


def enforceBdy(S):
    ''' Enforces the boundary conditions on S, the temperature values on the domain's grid points'''
    S[:, 0] = 1
    S[:, -1] = 1
    S[0, :] = 1
    S[-1, :] = 1
    return S


S = enforceBdy(S)


def Laplace(T):
    '''Computes the Laplacian operator, del-squared on the temperature data'''
    tmp_x, tmp_y = np.gradient(T, dx, dy)
    tmp_x, _ = np.gradient(tmp_x, dx)
    _, tmp_y = np.gradient(tmp_y, dy)
    return tmp_x+tmp_y


# iteration
nmax = int(tmax/dt)
for n in range(nmax):
    dSdt = alpha*Laplace(S)
    S = S + dSdt*dt
    S = enforceBdy(S)
    if n % 100 == 0:
        noise = np.random.normal(size=S.shape)*.1
        SimulateData.append(S.copy()+noise)
        SimulateDataNoNoise.append(S.copy())
    # check for convergence
    err = np.abs(dSdt*dt).max()
    if err <= epsilon:
        break

for i, frame in enumerate(SimulateData):
    plt.imshow(frame, vmin=0, vmax=1, cmap='inferno')
    plt.title(f'Frame {i+1}')
    plt.show()

# Creates Tensor
X = np.stack(SimulateData, 2)
nx,ny,nt = X.shape
# CP Decomposition
err = []
for i in range(1,11):
    CP_Heat = tl.decomposition.parafac(X,i)
    reconstructed = tl.kruskal_to_tensor(CP_Heat)  
    err.append(((X-reconstructed)**2).sum())
AIC = [2*e + 2*(i+1) for i,e in enumerate(err)]

idxmin = np.argmin(AIC)
R = idxmin+1
min_AIC = AIC[idxmin]
plt.plot(np.arange(1,11),AIC)
plt.xlabel('R')
plt.ylabel('AIC')
plt.show()
P = tl.decomposition.parafac(X,R)
plt.figure()
plt.plot(P[1][2])
plt.xlabel('Time/frame')
plt.legend(['Component ' +str(i) for i in range(1,R+1)])
plt.show()

for i in range(idxmin+1):
    A = P[1][0][:,i]
    B = P[1][1][:,i]
    XY = np.outer(A,B)
    plt.imshow(XY,cmap='inferno')
    plt.title(f'Spatial Component {i}x{i}')
    plt.show()

# Tucker Decomposition
AIC = {}
for i,j,k in product(range(1,5),repeat=3):
    tucker_heat = tl.decomposition.tucker(X,[i,j,k])
    reconstructed = tl.tucker_to_tensor(tucker_heat)  
    e = ((X-reconstructed)**2).sum()
    n_params = i*j*k+np.prod(tucker_heat[1][0].shape)+np.prod(tucker_heat[1][1].shape)+np.prod(tucker_heat[1][2].shape)
    AIC[(i,j,k)] = 2*e + 2*n_params

minIJK = min(AIC,key=AIC.get)
# minIJK = (3,3,3)
minAIC = AIC[minIJK]
tucker = tl.decomposition.tucker(X,minIJK)
I,J,K = minIJK
plt.figure()
plt.plot(tucker[1][-1])
plt.xlabel('Time/frame')
plt.legend(['Component ' +str(i) for i in range(1,K+1)])
plt.show()
for i,j in product(range(I),range(J)):
    A = tucker[1][0][:,i]
    B = tucker[1][1][:,j]
    XY = np.outer(A,B)
    plt.imshow(XY,cmap='inferno')
    plt.title(f'Spatial Component {i+1}x{j+1}')
    plt.show()

