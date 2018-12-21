import cv2
import numpy as np

img = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)

# Get the BGR value at specific pixel
px = img[100, 500]
print(px)

# Get the BGR attributes for pixel within region of interest
roi = img[100:150, 100: 150]
print(roi)

left = 436
right = 652
top = 148
bottom = 430

# cv2 pixel array is indexed by (x, y) instead of (y, x)
frog = img[top:bottom, left:right]


# cv2.imshow('frog', frog)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(np.array(img).shape)
print(np.array(frog).shape)
img[0:bottom - top, 0:right - left] = frog

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

