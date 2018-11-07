import cv2
import numpy as np
import os

os.chdir('./data')

img1 = cv2.imread('img1.jpg')
img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25)

img2 = cv2.imread('img2.jpg')
img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)

img2 = cv2.imread('img2.jpg')
img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)

img3 = cv2.imread('background.jpg')
img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)

rows, cols, channels = img3.shape

roi = img1[0:rows, 0:cols]

img3gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img3gray, 220, 225, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img3_fg = cv2.bitwise_and(img3, img3, mask=mask)

dst = cv2.add(img1_bg, img3_fg)
img1[0:rows, 0:cols] = dst


# overlap the two images without losing opaqueness
add = img1 + img2

# (b1, g1, r1) + (b2, g2, r2) -> built-in add max capped at 255
# add = cv2.add(img1, img2)

# weighted imposing one image on top of another
weighted = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)


cv2.imshow('res', img1)
# cv2.imshow('add', add)
# cv2.imshow('weighted', weighted)
# cv2.imshow('mask', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()








