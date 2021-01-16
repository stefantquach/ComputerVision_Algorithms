## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Fill out these functions yourself
def im2wv(img,nLev):
    output = []
    kernels = [[[0.5,0.5],[0.5,0.5]], [[-0.5,-0.5],[0.5,0.5]], [[-0.5,0.5],[-0.5,0.5]], [[0.5,-0.5],[-0.5,0.5]]]

    for i in range(nLev):
        level = [] # List to store H1, H2, H3
        # Calculating H1, H2, H3
        for kernel in kernels[1:]:
            k = np.pad(kernel, ((1,0),(1,0)))
            H = conv2(img,k[::-1,::-1] , mode='same', boundary='fill')
            level.append(H[::2,::2])

        # Calculating L
        k = np.pad(kernels[0], ((1,0),(1,0)))
        img = conv2(img,k[::-1,::-1] , mode='same', boundary='fill')[::2,::2]

        output.append(level)

    output.append(img)
    return output


def wv2im(pyr):
    img = pyr[-1]
    # This is the same coordinate transform to get the wavelet
    # but the multiplication requires k to be transposed
    k = 0.5*np.array([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]])
    for level in reversed(pyr[:-1]):
        stacked = np.stack([img]+level, axis=2)
        mult = np.matmul(stacked, k) #

        img = np.zeros(2*np.array(img.shape))
        for i in range(4):
            img[i%2::2, int(i/2)::2] = mult[:,:,i]

    return img


# Fill this out
# You'll get a numpy array/image of coefficients y
# Return corresponding coefficients x (same shape/size)
# that minimizes (x - y)^2 + lmbda * abs(x)
def denoise_coeff(y,lmbda):
    x  = y - 0.5*lmbda
    x1 = y + 0.5*lmbda

    cost = lambda x: (x-y)**2 +lmbda*np.abs(x)
    ans = np.where(cost(x) < cost(x1), x, x1)
    ans = np.where(cost(0) < cost(ans), 0, ans)
    return ans



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program

lmain = 0.88

img = np.float32(imread(fn('inputs/p1.png')))/255.

pyr = im2wv(img,4)
for i in range(len(pyr)-1):
    for j in range(2):
        pyr[i][j] = denoise_coeff(pyr[i][j],lmain/(2**i))
    pyr[i][2] = denoise_coeff(pyr[i][2],np.sqrt(2)*lmain/(2**i))

im = wv2im(pyr)
imsave(fn('outputs/prob1.png'),clip(im))
