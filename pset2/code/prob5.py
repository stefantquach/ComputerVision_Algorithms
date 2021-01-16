## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement using conjugate gradient, with a weight = 0 for mask == 0, and proportional
# to n_z^2 elsewhere. See slides.

def ntod(nrm, mask, lmda):
    iter = 100 # number of iterations, should be 100 for final version
    w = np.where(mask, nrm[:,:,2]**2,0)
    N = mask.shape[0]*mask.shape[1]

    # Constants
    gx = -nrm[:,:,0]/nrm[:,:,2]
    gx = np.where(mask, gx, 0)
    gy = -nrm[:,:,1]/nrm[:,:,2]
    gy = np.where(mask, gy, 0)
    fx = np.array([[0.5, 0, -0.5]])
    fy = np.array([[0.5], [0], [-0.5]])
    fr = np.array([[-1/9, -1/9, -1/9],[-1/9, 8/9, -1/9],[-1/9, -1/9, -1/9]])

    conv2_ = lambda x,y: conv2(x, y, mode='same', boundary='fill') # lambda for convolution
    b = conv2_(gx*w, fx[::-1,::-1]) + conv2_(gy*w, fy[::-1,::-1])

    # lambda for calculating Qp. Note: fr flipped is still fr
    Qp = lambda p: conv2_(conv2_(p, fx)*w, fx[::-1,::-1]) + conv2_(conv2_(p, fy)*w, fy[::-1,::-1]) + lmda*conv2_(conv2_(p, fr), fr)

    # Initial values
    Z = np.zeros(mask.shape) # initial guess is all zeros
    r=b # r0=b-QZ0, since Z0 is zero, then its just b
    p=r

    for i in range(iter):
        alpha = np.sum(r**2)/np.sum(p*Qp(p))
        Z = Z + alpha*p
        # print(p)
        r_ = r - alpha*Qp(p) # r_k+1
        beta = np.sum(r_**2)/np.sum(r**2)
        r = r_
        p = r_ + beta*p

    return Z


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-7)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
