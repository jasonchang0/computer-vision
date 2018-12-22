import cv2
import numpy as np
import os

os.chdir('../data')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

loop = True
while loop:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 5)
        for (eyeX, eyeY, eyeW, eyeH) in eyes:
            cv2.rectangle(roi_color, (eyeX, eyeY), (eyeX + eyeW, eyeY + eyeH), (0, 255, 0), thickness=2)

    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        loop = not loop

# Release the task assigned to camera
cv2.destroyAllWindows()
cap.release()





