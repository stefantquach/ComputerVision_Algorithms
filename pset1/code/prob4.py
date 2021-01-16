## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i):
    Y = np.zeros(X.shape) #  output image
    normalize = np.zeros(X.shape) # to normalize at the end

    # Spatial part is just a gaussian kernel and does not change pixel to pixel
    # modified from https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php
    ax = np.linspace(-K,K, num=2*K+1)
    x,y = np.meshgrid(ax,ax) # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    d2 = (x**2+y**2) # calculating squared distance
    spatial = np.exp(-0.5 * d2/sgm_s**2)

    # image padded with zeros in non-color dimension for filtering
    padded = np.pad(X, ((K,K),(K,K),(0,0)), mode="constant")

    # Calculating weights. This is vertorized to calculate the weight for each
    # pixel at the same time. The output value is also computed during this time
    for shift_y in ax:
        for shift_x in ax:
            endpoint_x = -K+int(shift_x)
            endpoint_y = -K+int(shift_y)
            if(-K+int(shift_x) == 0):
                endpoint_x = None
            if(-K+int(shift_y) == 0):
                endpoint_y = None

            pad = padded[K+int(shift_x):endpoint_x, K+int(shift_y):endpoint_y, :]
            exponent = np.sum((X-pad)**2, axis=2)
            B = spatial[K+int(shift_x), K+int(shift_y)]*np.exp(-0.5*exponent/sgm_i**2)

            Y += pad*np.stack([B,B,B], axis=2) # stacking to match dimentions
            normalize += np.stack([B,B,B], axis=2)

    return Y/normalize


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


img1 = np.float32(imread(fn('inputs/p4_nz1.jpg')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.jpg')))/255.

K=9

print("Creating outputs/prob4_1_a.jpg")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.jpg'),clip(im1A))


print("Creating outputs/prob4_1_b.jpg")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.jpg'),clip(im1B))

print("Creating outputs/prob4_1_c.jpg")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.jpg'),clip(im1C))

# Repeated application
print("Creating outputs/prob4_1_rep.jpg")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.jpg'),clip(im1D))

# Try this on image with more noise
print("Creating outputs/prob4_2_rep.jpg")
im2D = bfilt(img2,2,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.jpg'),clip(im2D))
