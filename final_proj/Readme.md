# CSE559A Final Project
Implements parametric and non-parametric approaches to background subtraction.

## File outlines
* `bg_subtract.py`: implements the non-parametric approach from Elgammal et al.
* `parametric_sub.py`: implements the mixed Gaussian method approach from Stauffer et al. NOTE: This is only implemented for grayscale images.
* `webcam_test.py`: This file runs the code for either background subtractor. A video file can be inputted or the webcam will be used.

## Report
A report can be found in `~/project`.

References:  
* A. Elgammal, D. Harwood, and L. Davis.  Non-parametricmodel  for  background  subtraction.    In  D.  Vernon,  editor,Computer  Vision  —  ECCV  2000,  pages  751–767,  Berlin,Heidelberg, 2000. Springer Berlin Heidelberg
* C.  Stauffer  and  W.  E.  L.  Grimson.   Adaptive  backgroundmixture models for real-time tracking. InProceedings. 1999IEEE Computer Society Conference on Computer Vision andPattern  Recognition  (Cat.  No  PR00149),  volume  2,  pages246–252 Vol. 2, 1999.