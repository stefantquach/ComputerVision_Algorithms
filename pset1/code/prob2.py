## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input 8-bit grayscale image
# Return equalized image with intensities from 0-255
def histeq(X):
    # Generating CDF
    unique,unique_counts = np.unique(X, return_counts=True)
    counts = np.zeros(np.iinfo(np.uint8).max+1)
    counts[unique] = unique_counts
    cdf = np.cumsum(counts/np.size(X))

    # Applying equalization
    return cdf[X]*255.0


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/p2_inp.jpg'))

out = histeq(img)

out = np.maximum(0,np.minimum(255,out))
out = np.uint8(out)

imsave(fn('outputs/prob2.jpg'),out)
