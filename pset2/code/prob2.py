## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

## Take color image, and return 'white balanced' color image
## based on gray world, as described in Problem 2(a). For each
## channel, find the average intensity across all pixels.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2a(img):
    alpha = 1/np.mean(img, axis=(1,0))
    sum = 1/3*np.sum(alpha)
    alpha /= sum

    return img*alpha


## Take color image, and return 'white balanced' color image
## based on description in Problem 2(b). In each channel, find
## top 10% of the brightest intensities, take their average.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2b(img):
    a = img.reshape(img.shape[0]*img.shape[1],3)
    sorted = np.sort(a, axis=0)
    tresh = int(sorted.shape[0]*0.9)
    # tresh = 0

    mean = np.mean(sorted[tresh:,:], axis=0)
    print(mean.shape)
    alpha = 1/mean
    sum = 1/3*np.sum(alpha)
    alpha /= sum

    return img * alpha



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program
im1 = np.float32(imread(fn('inputs/CC/ex1.jpg')))/255.
im2 = np.float32(imread(fn('inputs/CC/ex2.jpg')))/255.
im3 = np.float32(imread(fn('inputs/CC/ex3.jpg')))/255.

#
# im1a = balance2a(im1)
# im2a = balance2a(im2)
# im3a = balance2a(im3)
#
# imsave(fn('outputs/prob2a_1.png'),clip(im1a))
# imsave(fn('outputs/prob2a_2.png'),clip(im2a))
# imsave(fn('outputs/prob2a_3.png'),clip(im3a))

im1b = balance2b(im1)
im2b = balance2b(im2)
im3b = balance2b(im3)

imsave(fn('outputs/prob2b_1.png'),clip(im1b))
imsave(fn('outputs/prob2b_2.png'),clip(im2b))
imsave(fn('outputs/prob2b_3.png'),clip(im3b))
