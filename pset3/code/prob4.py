## Default modules imported. Import more if you need to.

import numpy as np
import matplotlib.pyplot as plt

## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):
    N = pts.shape[0]
    src = pts[:,0:2]
    dst = pts[:,2:]
    # padding a 1 for homogenous coordinates
    src = np.pad(src, ((0,0),(0,1)), constant_values=1)
    dst = np.pad(dst, ((0,0),(0,1)), constant_values=1)

    # Helper values. These are just src*dst_{x,y,z}
    a = np.stack([dst[:,0],dst[:,0],dst[:,0]],axis=-1)*src
    b = np.stack([dst[:,1],dst[:,1],dst[:,1]],axis=-1)*src
    c = np.stack([dst[:,2],dst[:,2],dst[:,2]],axis=-1)*src

    # Creating A matrix
    A = np.zeros([3*N,9])
    A[0::3,3:6] = -c
    A[0::3,6:]  = b
    A[1::3,0:3] = c
    A[1::3,6:]  = -a
    A[2::3,0:3] = -b
    A[2::3,3:6] = a

    # Solving
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    h = vh[-1,:]
    # print(h)
    H = h.reshape(3,3)
    return H
    

# Perfomes bilinear interpolation. Assumes pts is a 2xN array
# Returns channel x N array with intensities
def bilinear_interp(img, pts):
    x = pts[0,:]
    y = pts[1,:]
    fl_x = np.floor(x).astype(int)
    fl_y = np.floor(y).astype(int)
    w_x = x-fl_x
    w_y = y-fl_y
    w_x = np.stack([w_x, w_x, w_x], axis=-1)
    w_y = np.stack([w_y, w_y, w_y], axis=-1)

    res = w_y*(w_x*img[fl_y, fl_x] + (1-w_x)*img[fl_y, fl_x+1])
    res += (1-w_y)*(w_x*img[fl_y+1, fl_x] + (1-w_x)*img[fl_y+1, fl_x+1])
    return res

    


# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):
    # Getting homography
    H = src.shape[0]
    W = src.shape[1]
    print(H,W)
    spts = np.array([[0,0],[W-1,0],[0, H-1],[W-1,H-1]])
    H_ = getH(np.concatenate([dpts, spts], axis=1))

    ## Splicing image into dest
    # Larger rectangle encompassing quadralateral
    llx = min(dpts[:,0])
    lly = min(dpts[:,1])
    rrx = max(dpts[:,0])
    rry = max(dpts[:,1])

    x,y = np.meshgrid(np.arange(llx, rrx+1), np.arange(lly, rry+1))
    n_pts = int((rrx-llx+1)*(rry-lly+1))
    pts = np.stack([x.reshape(n_pts), y.reshape(n_pts), np.ones(n_pts)], axis=0) # all possible points in quad
    
    # Computing coordinates in source
    src_pts = H_ @ pts
    src_pts = src_pts[:2,:]/np.stack([src_pts[2,:],src_pts[2,:]],axis=0)

    # Filtering out of bounds points 
    check_x = np.logical_and(0 <= src_pts[0,:], src_pts[0,:] <= W-1)
    check_y = np.logical_and(0 <= src_pts[1,:], src_pts[1,:] <= H-1)
    in_idx = np.all(np.stack([check_x, check_y], axis=0), axis=0) 
    src_pts = src_pts[:,in_idx]
    pts = pts[:2,in_idx].astype(int)

    # splicing and bilinear interp
    dest[pts[1,:], pts[0,:]] = bilinear_interp(src, src_pts)
    return dest
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

# Test code for getH()
# p1 = np.array([[0, 0], [100, 0], [100, 200], [0, 200]])
# p2 = np.array([[207, 153], [385, 132], [386, 253], [206, 232]])
# pts = np.concatenate([p1,p2], axis=1)
# src = np.pad(p1, ((0,0),(0,1)), constant_values=1)
# print(getH(pts))
# p_ = getH(pts) @ np.transpose(src)
# p_ = p_[:2,:]/np.stack([p_[2,:],p_[2,:]],axis=0)
# plt.scatter(p1[:,0],p1[:,1])
# plt.scatter(p_[0,:],p_[1,:])
# plt.scatter(p2[:,0],p2[:,1])
# plt.show()

simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/prob4.png'),comb)
