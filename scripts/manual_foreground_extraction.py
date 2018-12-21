import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('../data')

img = cv2.imread('foreground_extraction.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# third dimension of the image is color channels
mask = np.zeros(img.shape[:2], np.uint8)

bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)

fgL = 400
fgR = 700

fgU = 250
fgD = 600

rect = tuple(np.array([fgL * .99, fgU * .99, fgR * 1.01, fgD * 1.01]).astype(int))
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

'''
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask[:, :, np.newaxis]
'''

# newmask is the manually labelled mask image
newmask = cv2.imread('newmask.png', cv2.IMREAD_GRAYSCALE)

# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1

mask, bgModel, fgModel = cv2.grabCut(img, mask, None, bgModel, fgModel, 15, cv2.GC_INIT_WITH_MASK)

mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask[:, :, np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()






