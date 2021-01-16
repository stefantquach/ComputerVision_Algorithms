# WUSTL CSE559A: Computer Vision Problem sets
All the following folders contain problem sets from the course, CSE559A Computer vision, at Washington University in St. Louis and taught by Ayan Charkrabarti. The topics and algorithms covered in each problem set are outlined below.

## Problem sets
### Problem Set 0
Basic environment setup, flipping an image.

### Problem Set 1
* Image modeling with noise
* Histogram Equalization
* Image gradients, edge detection
* Bilateral filtering
* Fourier Transforms
* Harr Wavelet decomposition and recomposition

### Problem Set 2
* Wavelet Regularization
* White Balance (multiple approaches)
* Photometric stereo
* Depth map generation using the Frankot-Chellappa method
* Depth map generation using the conjugate gradient method

### Problem Set 3
* Homographies and correspondences
* Robust line fitting (iterative, RANSAC)
* Camera Stereo properties
* Homography to splice one image into another
* Census transform and cost volumes (disparity maps)

### Problem Set 4 
* Stereo matching
* Bilateral filtering on cost volumes
* Viterbi algorithm, Semi-global matching
* Lucas-Kanade optical flow

### Problem Set 5
* Clustering using Simple Linear Iterative Clustering (SLIC)
* Autograd system
    * Adding momentum
    * Convolutional layer

Note: The convolution layer implemented does not implement strides. The version with strides was not finished.

## Final Project
Implementing background subtraction using parametric and non-parametric methods. The report can be found in the `project` folder.  

References:  
* A. Elgammal, D. Harwood, and L. Davis.  Non-parametricmodel  for  background  subtraction.    In  D.  Vernon,  editor,Computer  Vision  —  ECCV  2000,  pages  751–767,  Berlin,Heidelberg, 2000. Springer Berlin Heidelberg
* C.  Stauffer  and  W.  E.  L.  Grimson.   Adaptive  backgroundmixture models for real-time tracking. InProceedings. 1999IEEE Computer Society Conference on Computer Vision andPattern  Recognition  (Cat.  No  PR00149),  volume  2,  pages246–252 Vol. 2, 1999.