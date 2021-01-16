from parametric_sub import *
from bg_subtract import KDE_subtractor
import numpy as np
import cv2
import time

# cap = cv2.VideoCapture('highway.mp4')
cap = cv2.VideoCapture(0)
_,frame = cap.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
H,W = frame.shape
frame = frame.astype(np.float64)


K = 3
T = 0.5
alpha = 0.01

# subtract = mixed_gaussian_subtractor(K,T, alpha, frame)
subtract = KDE_subtractor(8, 0.01, (H,W,1))

count = 0
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype(np.float64)
    frame = frame.reshape(H,W,1)
    # print(frame)
    fg_mask = subtract.filter_frame(frame)
    # fg_mask = subtract.filter_frame(frame, th2=0.01, k_shape=(5,5))
    
    frame = frame.astype(np.uint8)
    # fg_mask = np.logical_not(fg_mask)
    fg_mask = fg_mask.astype(np.uint8)
    # print(fg_mask)
    # fg_img = np.where(fg_mask, frame, 0)
    # print((fg_mask==False).all())


    cv2.imshow('foreground', fg_mask*255)
    cv2.imshow('original', frame)

    # if count==20:
    #     cv2.imwrite('mg_highway_img.jpg', frame)
    #     cv2.imwrite('mg_highway_fg.jpg', fg_mask*255)

    count += 1
    print(count)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# time.sleep(10)
cap.release()
cv2.destroyAllWindows()