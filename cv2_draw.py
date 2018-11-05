import cv2
import numpy as np

img = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)

# CV2 uses BGR color space
cv2.line(img, (0, 0), (150, 150), (255, 255, 255), 15)
cv2.rectangle(img, (15, 25), (200, 150), (0, 255, 0), 10)
cv2.circle(img, (100, 50), 20, (0, 0, 255), -1)

# Draw polygons given coordinates of the vertices
pts = np.array([[10, 5], [20, 20], [30, 30], [50, 10], [100, 160]], dtype=np.int32)
pts = pts.reshape((-1, 1, 2))

# print(list(pts))
# print([pts])

cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=3)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV Textual Test!', (400, 500), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('image', img)

# Waits for any processed key for delay of n milliseconds
cv2.waitKey(0)
cv2.destroyAllWindows()





