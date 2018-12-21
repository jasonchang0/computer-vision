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
    lower_red = np.array([150, 150, 150])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([0, 150, 150])
    upper_red = np.array([30, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = cv2.bitwise_or(mask1, mask2)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((15, 15), np.float32) / (15 * 15)
    smooth = cv2.filter2D(res, -1, kernel=kernel)

    blur = cv2.GaussianBlur(res, (15, 15), 0)
    median = cv2.medianBlur(res, 15)
    bilateral = cv2.bilateralFilter(res, 15, 75, 75)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('result', res)

    # cv2.imshow('smooth', smooth)
    # cv2.imshow('blur', blur)
    # cv2.imshow('median', median)
    # cv2.imshow('bilateral', bilateral)

    k = cv2.waitKey(1) & 0xFF
    loop = (k != ord('q')) and loop


cv2.destroyAllWindows()
cap.release()


