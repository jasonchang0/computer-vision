import cv2
import numpy as np
import os


os.chdir('../data')

img = cv2.imread('corner_sample.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')

'''
cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, 
minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) 
→ corners
'''
corners = cv2.goodFeaturesToTrack(gray, 100, 0.25, 25).astype(int)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, (0, 255, 255), -1)


cv2.imshow('Corner', img)

cv2.waitKey(0)
cv2.destroyAllWindows()




