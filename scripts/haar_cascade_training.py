import cv2
import numpy as np
import os

os.chdir('../data')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
watch_cascade = cv2.CascadeClassifier('haarcascade_watch.xml')

cap = cv2.VideoCapture(0)

loop = True
while loop:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 3, 5)
    watches = watch_cascade.detectMultiScale(gray, 2, 5)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, w, h) in watches:
        cv2.putText(frame, 'Watch', (x - w, y - h), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (int(x + 1.05 * w), y), int(max(w, h)), (255, 255, 0), 2)

    for (x, y, w, h) in faces:
        cv2.putText(frame, 'Face', tuple(np.array([x - 0.1 * w, y - 0.1 * h]).astype(int)), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 3, 5)

        for (eyeX, eyeY, eyeW, eyeH) in eyes:
            # print((eyeX, eyeY, eyeW, eyeH))
            cv2.putText(frame, 'Eye', tuple(np.array([x + eyeX - 0.1 * eyeW, y + eyeY - 0.1 * eyeH]).astype(int)),
                        font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.rectangle(roi_color, (eyeX, eyeY), (eyeX + eyeW, eyeY + eyeH), (0, 255, 0), thickness=2)

    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        loop = not loop

# Release the task assigned to camera
cv2.destroyAllWindows()
cap.release()









