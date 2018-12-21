import cv2
import numpy as np

cap = cv2.VideoCapture(0)

loop = True
while loop:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # hsv -> hue | sat | value
    # lower_blue = np.array([0,120,120])
    # upper_blue = np.array([80,255,255])
    #
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    # only for red -30 to 0 and 0 to 30 -  2masks
    lower_red = np.array([160, 180, 180])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([0, 180, 180])
    upper_red = np.array([20, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = cv2.bitwise_or(mask1, mask2)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((3, 3), np.uint8)

    # narrows the roi by eliminating outliers
    erosion = cv2.erode(mask, kernel, iterations=1)
    res = cv2.bitwise_and(res, res, mask=erosion)

    # expands the roi by assimilating gaps
    dilation = cv2.dilate(mask, kernel, iterations=1)
    # res = cv2.bitwise_and(res, res, mask=dilation)

    # removes false positive in the background (+ves)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel)
    res = cv2.bitwise_and(res, res, mask=opening)

    # removes false negative in the roi (-ves)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel)
    # res = cv2.bitwise_and(res, res, mask=closing)

    res = cv2.medianBlur(res, 5)

    '''
    # Convolution-based smoothing and blurring
    kernel = np.ones((10, 10), np.float32) / (15 * 15)
    smooth = cv2.filter2D(res, -1, kernel=kernel)

    # blur = cv2.GaussianBlur(res, (15, 15), 0)
    median = cv2.medianBlur(res, 15)
    # bilateral = cv2.bilateralFilter(res, 15, 75, 75)
    '''

    # difference between input image and Opening of the image
    # tophat = image - opening = image - (image - false + ves) = false + ves
    tophat = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel=kernel)
    # cv2.imshow('Tophat', tophat)

    # difference between input image and Closing of the image
    # blackhat = image - closing = image - (image - false - ves) = false - ves
    blackhat = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel=kernel)
    # cv2.imshow('BLackhat', blackhat)

    cv2.imshow('frame', frame)
    cv2.imshow('result', res)

    # cv2.imshow('erosion', erosion)
    # cv2.imshow('dilation', dilation)
    # cv2.imshow('opening', opening)
    # cv2.imshow('closing', closing)

    # cv2.imshow('smooth', smooth)
    # cv2.imshow('blur', blur)
    # cv2.imshow('median', median)
    # cv2.imshow('bilateral', bilateral)

    k = cv2.waitKey(1) & 0xFF
    loop = (k != ord('q')) and loop


cv2.destroyAllWindows()
cap.release()


