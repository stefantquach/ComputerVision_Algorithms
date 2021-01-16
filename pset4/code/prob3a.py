## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################
# shifts an array and pads with the value. offset is a arraylike 
# with (offset_y, offset_x). Downwards is a positive y shift
def shift_pad(img, offset, constant=0):
    res = np.roll(img, offset, axis=(0,1))
    # y direction
    if offset[0] > 0:
        res[:offset[0],:]=constant
    elif offset[0] < 0:
        res[offset[0]:,:]=constant
    # x direction
    if offset[1] > 0:
        res[:,:offset[1]]=constant
    elif offset[1] < 0:
        res[:,offset[1]:]=constant
    return res    


# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):
    W = img.shape[1]
    H = img.shape[0]
    c = np.zeros([H,W],dtype=np.uint32)

    # generating offset amounts
    offsets = [(x,y) for y in range(-2,3) for x in range(-2,3) if not x == 0 == y]
    for a,b in offsets:
        # using constant 500 since its bigger than anything in the image
        # Therefore anything outside results in a 0
        c = (c << 1) | (img > shift_pad(img, (a,b), constant=500)) 
    return c

# Copy this from solution to problem 2.
def buildcv(left,right,dmax):
    c_left = census(left)
    c_right = census(right)
    cv = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
    for i in range(dmax+1):
        if i==0:
            cv[:,:,i] = hamdist(c_left, c_right)
        else:
            cv[:,:,i] = hamdist(c_left, np.roll(c_right,i, axis=1))

    return cv


# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    d = np.zeros([H,W])
    z = np.zeros([H,W,D])
    C_bar = cv[:,0,:]
    for x in range(1, W):
        C_til = C_bar - np.min(C_bar, axis=1, keepdims=True)
        
        # Finding min C_til + S
        # creating shifted versions.
        p_2 = P2*np.ones(C_til.shape)
        plus_1 = shift_pad(C_til, (0,-1), constant=24)+P1
        minus_1= shift_pad(C_til, (0,1), constant=24)+P1

        compare = np.array([p_2, plus_1, minus_1, C_til])
        amin_compare = np.argmin(compare, axis=0)

        # calculating Z
        amin_ctil = np.tile(np.argmin(C_til, axis=1), (D,1)).transpose()
        ranges = np.tile(np.arange(D), (H,1)) 
        z[:,x,:] = np.where(amin_compare==0, amin_ctil, \
            np.where(amin_compare==1, shift_pad(ranges, (0,-1)), \
                np.where(amin_compare==2, shift_pad(ranges, (0,1)), \
                    np.where(amin_compare==3, ranges, amin_compare))))
        
        # Calculating next C_bar
        C_bar = np.min(compare, axis=0) + cv[:,x,:]

    # Reverse direction
    d[:,-1] = np.argmin(C_bar, axis=1)
    for x in reversed(range(0,W-1)):
        d[:,x] = z[np.arange(d.shape[0]),x+1,d[:,x+1].astype(int)]

    return d
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = viterbilr(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
