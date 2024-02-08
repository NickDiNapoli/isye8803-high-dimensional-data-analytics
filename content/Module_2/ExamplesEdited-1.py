import scipy.misc as spm
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import k_means
import cv2
import skfda
from skfda.representation.basis import BSpline
from PIL import Image, ImageOps

from skimage.filters import threshold_multiotsu
from skimage import io, color
np.random.seed(0)

# Nick's code from hw
img = io.imread('horse1-2.jpg')
img_gray = color.rgb2gray(img)*255

# Applying multi-Otsu threshold for the default value, generating
# three classes.
plt.figure(figsize=(16,16))
for level in range(2, 6):
    thresholds = threshold_multiotsu(img_gray, level)
    print(thresholds)
    regions = np.digitize(img_gray, bins=thresholds)
    plt.subplot(4, 1, level - 1)
    plt.imshow(regions, cmap='gray')
    plt.title(f"Otsu's method segmentation, level={level}")

plt.tight_layout()
plt.savefig("otsu.png")
plt.show()


'''
# %% Image Transformation
# Image Histogram
I = cv2.imread('albert.png')
plt.imshow(I)
n = 256
plt.figure()
colConv = np.array([0.2989, 0.5870, 0.1140])
Ig = I.astype(float)@colConv
bins = np.arange(0, 256, 1, dtype=int)
b, bins, patches = plt.hist(Ig.ravel(), bins=bins)
plt.xlim([0, n+1])
plt.show()

# Histogram Stretching
Img1 = ((Ig-np.min(Ig))/(np.max(Ig)-np.min(Ig))*205)
Image.fromarray(Img1.astype('uint8'))
plt.figure()
b, bins, patches = plt.hist(Img1.ravel(), bins=bins)
Img2 = ((Ig-np.min(Ig))/(np.max(Ig)-np.min(Ig))*150)
plt.figure()
Image.fromarray(Img2.astype('uint8'))
plt.figure()
b, bins, patches = plt.hist(Img2.ravel(), bins=bins)

# %% Image filtering and convolution

# * Image filtering
Y = plt.imread('albert.png')[:, :, :-1]
Y = np.round((Y*255)).astype(int)

Kernels = {
    'edgeDetector1': np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]),
    'edgeDetector2': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    'edgeDetector3': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    'sharpening':  np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'blurring1': np.ones((3, 3))/9,
    'blurring2': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
}
Yk = {}
for i, (name, filt) in enumerate(Kernels.items()):
    Yk[name] = ndi.convolve(Y.mean(2), filt)
    plt.imshow(Yk[name], cmap='gray', vmin=0, vmax=255)
    plt.title(name)
    plt.show()

#  * Denoising using splines
# 2D example: Generate data

def z(x, y):
    return 3*(1-x)**2.*np.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) - 1/3*np.exp(-(x+1)**2 - y**2)

n = 100
XX, YY = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
sigma = 0.5
peaks = z(XX, YY)
Y = peaks + np.random.normal(size=(n, n))*sigma
plt.figure()
plt.imshow(Y)
plt.title('Noisy Image')

x = np.linspace(1,100,100)
knots= np.linspace(1,100,10)
bs = BSpline(knots=knots, order=4)
b = bs(x).T
B=b.reshape(b.shape[1:3])
H = B@np.linalg.inv(B.T@B)@B.T
Yhat = H@Y@H
plt.figure()
plt.imshow(Yhat)
plt.title('After B-spline smoothing')

# %% IMAGE SEGMENTATION
# * Otsu's Method
# It's very strange, but matplotlib and MATLAB are reading different colors from the same image
# Step 1: get the histogram of image
Y = plt.imread('coin.png')[:, :, :-1]
Y = np.round((Y*255)).astype(int)
I = Y.ravel()
num_bins = 256
bins = np.arange(0, num_bins+1, 1, dtype=int)
hist, _ = np.histogram(I, bins=bins)
plt.figure()
plt.imshow(Y)
plt.title('Original Image')
# Step 2: Calculate group mean
p = hist / hist.sum()
omega = np.cumsum(p)
mu = np.cumsum(p * np.arange(1, num_bins+1))
mu_t = mu[-1]
# Step 3: find the maximum value of
sigma_b_squared = (mu_t * omega - mu)**2 / (omega * (1 - omega))
maxval = sigma_b_squared.max()
idx = np.argmax(sigma_b_squared)
# Step 4: Thresholding and get final image
level = (idx - 1) / (num_bins - 1)
BW = Y > level*num_bins
plt.figure()
plt.imshow(BW.astype(float))
plt.title("Otsu's Method segmentation")

# * K-means clustering 
# Load data
Y = plt.imread('coin.png')[:, :, :-1]
I = np.round((Y*255)).astype(int)
X= I.reshape(-1,I.shape[2]).astype(float)

# Set parameters
K=2 
max_iter = 100
# Clustering
N, d = X.shape
#  indicator matrix (each entry corresponds to the cluster of each point in X)
L = np.empty_like(X[:,0])
# centers matrix
C = X[np.random.choice(N,2,False),:].astype(float)
for i in range(max_iter):
    # step 1: optimize the labels
    dist = cdist(X,C)
    L = np.argmin(dist,1)
    # step 2: optimize the centers
    for k in range(K):
        C[k,:] = X[L==k,:].mean(0)
Y = L.reshape(I.shape[0],I.shape[1])
BW = Y.astype(float)
plt.figure()
plt.imshow(BW,cmap='gray')
plt.title('K-means segmentation')
plt.show()


# *  k-means clustering w/ Libraries
# input image
Y = plt.imread('CS.png')[:, :, :-1]
print(Y.shape)
I = np.round((Y*255)).astype(int)
plt.figure()
plt.imshow(I)
plt.title(f'Original Image') 

X=I.reshape(-1,I.shape[2]).astype(float)
print(X.shape)
plt.show()

# segmentation with different K values
Ks=[2, 3, 4, 5]
for k in Ks:
    centers, L,_ = k_means(X,k)
    Y = L.reshape(*I.shape[:2])
    plt.figure()
    plt.imshow(Y)
    plt.title(f'K-means, {k} Clusters')
    plt.show()


# %% EDGE DETECTION 
# * Sobel operator 
a = plt.imread('coin.png').mean(2)
m,n = a.shape

op = np.array([[1, 2, 1],[ 0, 0, 0],[-1, -2, -1]] )/8
x_mask = op.T
y_mask = op
fx = ndi.convolve(a,x_mask,mode='nearest')
fy = ndi.convolve(a,y_mask,mode='nearest')
plt.figure()
plt.imshow(fx)
plt.title('Sobel edge-detector, X-direction')
plt.figure()
plt.imshow(fy)
plt.title('Sobel edge-detector, Y-direction')

# %%
'''