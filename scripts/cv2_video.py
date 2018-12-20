import cv2
import numpy as np

# Use the system web camera as video source
cap = cv2.VideoCapture(0)

'''
fourcc = cv2.cv.CV_FOURCC(*'H264')
#or 
#fourcc = cv2.cv.CV_FOURCC(*'X264')
'''

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

loop = True
while loop:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out.write(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    '''
    0xFF is a hexadecimal constant which is 11111111 in binary. 
    By using bitwise AND (&) with this constant, it leaves only 
    the last 8 bits of the original (in this case, whatever cv2.waitKey(0) is).
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        loop = not loop

# Release the task assigned to camera
cap.release()
out.release()
cv2.destroyAllWindows()
