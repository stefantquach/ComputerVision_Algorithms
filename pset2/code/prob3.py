### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):
    imgs_ = np.array(imgs)
    gray = np.mean(imgs_, axis=3) # Converting to grayscale
    flat = gray.reshape(len(imgs), gray[0].size) # Flattening

    # Solving equations
    a = np.matmul(L.transpose(), L) # Coefficient a of ax = b
    b = np.matmul(L.transpose(), flat) # Coefficient b of ax = b
    N = np.linalg.solve(a, b) # this is 3 x n (n is number of pixels)
    N = N.transpose()         # n x 3 for reshaping
    N = N.reshape(imgs[0].shape) # HxWx3 array

    # Normalizing to unit vectors
    mag = np.linalg.norm(N, axis=2) # Calculate 2 norm of vectors for each pixel
    N = N / np.stack([mag,mag,mag], axis=2) # Stacking so element wise division works

    return np.where(np.stack([mask,mask,mask], axis=2), N,0)


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):
    K = len(imgs) # K
    n = imgs[0].shape[0]*imgs[0].shape[1] # H*W (total number of pixels)
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    # sum over k of l_k.' * l_k
    sum_l_k = np.sum(L**2)

    # n^T * n
    n_dot = np.sum(nrm**2, axis=2)
    den = n_dot*sum_l_k # denominator

    imgs = np.array(imgs)
    flat = nrm.reshape(n, 3) # flattening the image to an array of 3-vectors
    flat = flat.transpose()  # changing to 3 x n
    mat = np.dot(L, flat)
    mat = mat.reshape(K, H, W)
    num = imgs*(np.stack([mat, mat, mat], axis=-1))
    sum = np.sum(num, axis=0) # numerator

    return np.where(np.stack([mask,mask,mask], axis=2), sum / np.stack([den,den,den], axis=-1),0)

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
