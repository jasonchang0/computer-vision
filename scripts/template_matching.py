import cv2
import numpy as np
import os

os.chdir('../data')


'''
# use images as inputs
frame = cv2.imread('IMG_1041.jpg')
frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

temp = cv2.imread('matching_template.jpg', cv2.IMREAD_GRAYSCALE)
temp = cv2.resize(temp, (0, 0), fx=0.25, fy=0.25)

h, w = temp.shape

res = cv2.matchTemplate(frame_gray, temp, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for point in zip(*loc[::-1]):
    cv2.rectangle(frame, point, (point[0] + w, point[1] + h), (0, 255, 255), 2)

cv2.imshow('detected', frame)
'''

cap = cv2.VideoCapture(0)

loop = True
while loop:
    # use videos as inputs
    _, frame = cap.read()

    # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    temp = cv2.imread('matching_template.JPG', cv2.IMREAD_GRAYSCALE)
    temp = cv2.resize(temp, (0, 0), fx=0.25, fy=0.25)
    h, w = temp.shape

    res = cv2.matchTemplate(frame_gray, temp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    for point in zip(*loc[::-1]):
        cv2.rectangle(frame, point, (point[0] + w, point[1] + h), (0, 255, 255), 2)

    cv2.imshow('detected', frame)

    k = cv2.waitKey(1) & 0xFF
    loop = (k != ord('q')) and loop

cv2.destroyAllWindows()
cap.release()






