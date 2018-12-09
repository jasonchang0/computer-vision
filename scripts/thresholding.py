import cv2
import numpy as np
import os

os.chdir('./data')

img = cv2.imread('page.JPG')
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

retVal, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retVal2, threshold2 = cv2.threshold(imgGray, 180, 255, cv2.THRESH_BINARY)

retVal3, otsu = cv2.threshold(imgGray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

gaussian = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

cv2.imshow('original', img)

cv2.imshow('threshold', threshold)
cv2.imshow('threshold2', threshold2)

cv2.imshow('otsu', otsu)
cv2.imshow('gaussian', gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()






