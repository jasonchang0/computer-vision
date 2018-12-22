import cv2
import numpy as np
import os

os.chdir('../data')

cap = cv2.VideoCapture('sproul_timelapse.mov')
mog = cv2.createBackgroundSubtractorMOG2()

loop = True
while loop:
    _, frame = cap.read()
    fg_mask = mog.apply(frame)

    kernel = np.ones((5, 5), np.uint8)

    # narrows the roi by eliminating outliers
    erosion = cv2.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=erosion)

    kernel = np.ones((3, 3), np.uint8)

    # removes false positive in the background (+ves)
    opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel=kernel)
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=opening)

    # Convolution-based smoothing and blurring
    stride = 5
    kernel = np.ones((stride, stride), np.float32) / (stride ** 2)
    fg_mask = cv2.filter2D(fg_mask, -1, kernel=kernel)

    cv2.imshow('frame', frame)
    cv2.imshow('foreground', fg_mask)

    k = cv2.waitKey(1) & 0xFF
    loop = (k != ord('q')) and loop


cv2.destroyAllWindows()
cap.release()




