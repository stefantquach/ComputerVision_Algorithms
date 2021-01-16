## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2


## Fill out these functions yourself

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



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.jpg')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.jpg'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.jpg'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.jpg'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.jpg'),im)

# Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.jpg' % i),im)
