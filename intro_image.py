import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
# IMREAD_COLOR = 1
# IMREAD_UNCHANGED = -1

# Present image using cv2
# cv2 uses BGR for color space
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Present image using matplotlib
# matplotlib uses RGB for color space
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.plot([20, 40, 60, 80, 100], [40, 60, 80, 100, 120], color='c', linewidth=5)
# plt.show()

# To save image in the directory
cv2.imwrite('gray_sample.jpg', img)


