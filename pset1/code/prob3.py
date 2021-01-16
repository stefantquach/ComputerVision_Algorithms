## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
    #placeholder
    # Sobel X and Y operators
    D_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
    D_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    # Convolution in two directions
    dx = conv2(X, D_x, mode='same', boundary='symmetric')
    dy = conv2(X, D_y, mode='same', boundary='symmetric')
    # Calculating magnitude and
    H = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy,dx)

    return H,theta


def nms(E,H,theta):
    # rounding to nearest  pi/4
    theta /= np.pi/4
    theta = np.round(theta)*np.pi/4

    # # getting directions
    # x_dir = np.round(np.cos(theta))
    # y_dir = np.round(np.sin(theta))
    #
    # E_plus = E
    #
    # for i in range(1,H.shape[0]-1):
    #     for j in range(1,H.shape[1]-1):
    #         # check if edge
    #         if E[i,j]:
    #             if((H[i,j] <= H[int(i+y_dir[i,j]), int(j+x_dir[i,j])]) and
    #             (H[i,j] <= H[int(i+y_dir[i,j]), int(j+x_dir[i,j])])):
    #                 E_plus[i,j] = 0

    # alternate method without iteration
    # converting to [pi/2, -pi/2]
    theta = np.where(theta > np.pi/2, theta-np.pi, theta)
    theta = np.where(theta < -np.pi/2, theta+np.pi, theta)
    theta = np.where(theta == -np.pi/2, theta+np.pi, theta)

    # reference https://stackoverflow.com/questions/19878280/efficient-way-to-shift-2d-matrices-in-both-directions
    # rolling for horizontal and vertical axis
    H_y = (np.roll(H, -1, axis=0), np.roll(H, 1, axis=0))
    H_x = (np.roll(H, -1, axis=1), np.roll(H, 1, axis=1))
    # H_xy = (np.roll(np.roll(H, -1, axis=0), -1, axis=1), np.roll(np.roll(H, 1, axis=0), 1, axis=1))
    # H_yx = (np.roll(np.roll(H, 1, axis=0), -1, axis=1), np.roll(np.roll(H, -1, axis=0), 1, axis=1))
    H_xy = (np.roll(H, [-1,-1], axis=[0,1]), np.roll(H, [1,1], axis=[0,1]))
    H_yx = (np.roll(H, [1,-1], axis=[0,1]), np.roll(H, [1,-1], axis=[0,1]))

    angles = {0: H_x, np.pi/2: H_y, -np.pi/4:H_xy, np.pi/4:H_yx}
    for angle in angles:
        E_plus = np.where(np.logical_and(E, theta==angle), np.logical_and(H > angles[angle][0], H>angles[angle][1]), E)

    return E_plus

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.jpg')))/255.

H,theta = grads(img)

imsave(fn('outputs/prob3_a.jpg'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.jpg'),E0)
imsave(fn('outputs/prob3_b_1.jpg'),E1)
imsave(fn('outputs/prob3_b_2.jpg'),E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.jpg'),E0n)
imsave(fn('outputs/prob3_b_nms1.jpg'),E1n)
imsave(fn('outputs/prob3_b_nms2.jpg'),E2n)
