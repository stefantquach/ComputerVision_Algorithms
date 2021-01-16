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
#########################################


# Copy this from problem 2 solution.
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


# Do SGM. First compute the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then compute the disparity map as
# the argmin of the sum of these cost volumes. 
def SGM(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    agg_C_bar = np.zeros([H,W,D])

    # Left<-->Right 
    C_bar_f = np.zeros([H,W,D])
    C_bar_b = np.zeros([H,W,D])
    C_bar_f[:,0,:] = cv[:,0,:] # starting point forward
    C_bar_b[:,-1,:] = cv[:,-1,:] # starting point backwards
    for x in range(1, W-1):
        C_til_f = C_bar_f[:,x,:] - np.min(C_bar_f[:,x,:], axis=1, keepdims=True)
        C_til_b = C_bar_b[:,-x-1,:] - np.min(C_bar_b[:,-x-1,:], axis=1, keepdims=True)
        
        # Finding min C_til + S
        # creating shifted versions.
        p_2 = P2*np.ones(C_til_f.shape)
        plus_1_f = shift_pad(C_til_f, (0,-1), constant=24)+P1
        minus_1_f= shift_pad(C_til_f, (0,1), constant=24)+P1
        plus_1_b = shift_pad(C_til_b, (0,-1), constant=24)+P1
        minus_1_b= shift_pad(C_til_b, (0,1), constant=24)+P1

        compare_f = np.array([p_2, plus_1_f, minus_1_f, C_til_f])
        compare_b = np.array([p_2, plus_1_b, minus_1_b, C_til_b])

        # Calculating next C_bar
        C_bar_f[:,x+1,:] = np.min(compare_f, axis=0) + cv[:,x,:]
        C_bar_b[:,-x-2,:] = np.min(compare_b, axis=0) + cv[:,-x-1,:]

    agg_C_bar += C_bar_f + C_bar_b

    # Up <--> Down
    C_bar_f = np.zeros([H,W,D])
    C_bar_b = np.zeros([H,W,D])
    C_bar_f[0,:,:] = cv[0,:,:] # starting point forward
    C_bar_b[-1,:,:] = cv[-1:,:] # starting point backwards
    for y in range(1, H-1):
        C_til_f = C_bar_f[y,:,:] - np.min(C_bar_f[y,:,:], axis=1, keepdims=True)
        C_til_b = C_bar_b[-y-1,:,:] - np.min(C_bar_b[-y-1,:,:], axis=1, keepdims=True)
        
        # Finding min C_til + S
        # creating shifted versions.
        p_2 = P2*np.ones(C_til_f.shape)
        plus_1_f = shift_pad(C_til_f, (0,-1), constant=24)+P1
        minus_1_f= shift_pad(C_til_f, (0,1), constant=24)+P1
        plus_1_b = shift_pad(C_til_b, (0,-1), constant=24)+P1
        minus_1_b= shift_pad(C_til_b, (0,1), constant=24)+P1

        compare_f = np.array([p_2, plus_1_f, minus_1_f, C_til_f])
        compare_b = np.array([p_2, plus_1_b, minus_1_b, C_til_b])

        # Calculating next C_bar
        C_bar_f[y+1,:,:] = np.min(compare_f, axis=0) + cv[y,:,:]
        C_bar_b[-y-2,:,:] = np.min(compare_b, axis=0) + cv[-y-1,:,:]

    agg_C_bar += C_bar_f + C_bar_f
    return np.argmin(agg_C_bar ,axis=2)

    
    
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
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3b.jpg'),dimg)
