## Default modules imported. Import more if you need to.
### Problem designed by Abby Stylianou

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.
    # cluster_centers = np.zeros((num_clusters,2),dtype='int')

    # Convenient constants
    H = im.shape[0]
    W = im.shape[1]
    K = num_clusters

    ### calculate gradient
    grad = get_gradients(im)

    # Calculating S and creating evenly spaced grid
    S = int(np.sqrt(W*H/K))
    x = np.arange(S//2, W, step=S)
    y = np.arange(S//2, H, step=S)
    xv, yv = np.meshgrid(x, y)
    flat_x = xv.flatten()
    flat_y = yv.flatten()

    # shifting and calculating min in 3x3 area around centers
    shift_values = np.array([-1,0,1])
    shifted_stack = [grad[flat_y+i,flat_x+j] for i in range(-1,2) for j in range(-1, 2)]
    stack = np.stack(shifted_stack, axis=1)
    index = np.argmin(stack, axis=1)
    flat_x += shift_values[index % 3]
    flat_y += shift_values[index // 3]

    return np.stack([flat_x, flat_y], axis=1)

def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    L = np.zeros((h,w))
    S = int(np.sqrt(h*w/num_clusters))
    # Parameter to change
    alpha = 2
    num_iterations = 10

    # Creating augmented image
    spatial = np.stack(np.meshgrid(np.arange(h), np.arange(w)), axis=2)
    im_ = np.concatenate([im, alpha*spatial], axis=2)

    # Starting mean vectors
    u = im_[cluster_centers[:,1], cluster_centers[:,0]]

    # updating min_dist
    for _ in range(num_iterations):
        min_dist = np.inf*np.ones(L.shape) # min_dist starts at inf
        # Step 2
        for k in range(num_clusters):
            u_k = u[k,:]
            # getting indices of 2S x 2S square around u
            x = np.arange(u_k[3]/alpha-S if u_k[3]/alpha-S>=0 else 0, u_k[3]/alpha+S if u_k[3]/alpha+S < w else w).astype(int)
            y = np.arange(u_k[4]/alpha-S if u_k[4]/alpha-S>=0 else 0, u_k[4]/alpha+S if u_k[4]/alpha+S < h else h).astype(int)
            xv, yv = np.meshgrid(x,y)
            x = xv.flatten()
            y = yv.flatten()
            
            # Calculating distances
            u_k = u_k.reshape(1,5)
            dist = np.sum((im_[y,x] - u_k)**2, axis=1)
            L[y,x] = np.where(dist < min_dist[y,x], k, L[y,x])
            min_dist[y,x] = np.where(dist < min_dist[y,x], dist, min_dist[y,x])

        # Step 1    
        for k in range(num_clusters):
            u[k] = np.mean(im_[np.where(L==k)], axis=0)

    return L

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))

num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
