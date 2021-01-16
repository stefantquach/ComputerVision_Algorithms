## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from numpy.fft import fft2, ifft2

## Fill out these functions yourself

# From Pset 1
def kernpad(K,size):
    Ko = np.zeros(size,dtype=np.float32)

    # placing Kernel in top left corner
    Ko[0:K.shape[0], 0:K.shape[1]] = K

    # Shifting diagonally circularly to place center of kernel at 0,0
    y_shift = K.shape[0]//2
    x_shift = K.shape[1]//2
    Ko = np.roll(Ko, [-y_shift, -x_shift], axis=[0,1])

    return Ko

# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement in Fourier Domain / Frankot-Chellappa
def ntod(nrm, mask, lmda):
    # Getting standard domain variables
    gx = -nrm[:,:,0]/nrm[:,:,2]
    gx = np.where(mask, gx, 0)
    gy = -nrm[:,:,1]/nrm[:,:,2]
    gy = np.where(mask, gy, 0)

    fx = kernpad(np.array([[0.5, 0, -0.5]]), gx.shape)
    fy = kernpad(np.array([[0.5], [0], [-0.5]]), gx.shape)

    fr = [[-1/9, -1/9, -1/9],[-1/9, 8/9, -1/9],[-1/9, -1/9, -1/9]]
    fr = kernpad(np.array(fr), gx.shape)

    # Fourier transform
    Gx = fft2(gx)
    Gy = fft2(gy)
    Fx = fft2(fx)
    Fy = fft2(fy)
    Fr = fft2(fr)

    abs2 = lambda x: (np.conj(x)*x)

    Fz = (np.conj(Fx)*Gx + np.conj(Fy)*Gy)/(abs2(Fx) + abs2(Fy) + lmda*abs2(Fr)+1e-12)
    Fz[0,0] = 0

    return np.real(ifft2(Fz))


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
Z = ntod(nrm,mask,1e-6)


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
