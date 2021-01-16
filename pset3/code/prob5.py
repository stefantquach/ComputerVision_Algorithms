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





## Fill out these functions yourself

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
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):
    c_left = census(left)
    c_right = census(right)
    cost_volume = np.zeros([left.shape[0],left.shape[1],dmax+1])
    for i in range(dmax+1):
        if i==0:
            cost_volume[:,:,i] = hamdist(c_left, c_right)
        else:
            cost_volume[:,:,i] = hamdist(c_left, np.roll(c_right,i, axis=1))
            cost_volume[:,:i,i] = 24 # 24 is max hamdist
    
    disparity = np.argmin(cost_volume, axis=2)

    return disparity
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

# Testing census transform
# c_left = census(left)
# c_left = c_left*1/0x00FFFFFF
# c_left *= 255
# print(c_left)
# c_left = c_left.astype(int)
# imsave(fn('outputs/test.png'),c_left)

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
