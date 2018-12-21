import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir('../data')

img1 = cv2.imread('match1.JPG', cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)

img2 = cv2.imread('match2.JPG', cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

'''
Class implementing the ORB (oriented BRIEF) 
keypoint(kp) detector and descriptor(des) extractor.
'''
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

'''
Brute Force Matching using Hamming distance,
which measures the minimum number of substitutions 
required to change one string into the other.
'''
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

'''
Possible flags bit values are defined by DrawMatchesFlags

struct DrawMatchesFlags
{
    enum
    {
        DEFAULT = 0, // Output image matrix will be created (Mat::create),
                     // i.e. existing memory of output image may be reused.
                     // Two source images, matches, and single keypoints
                     // will be drawn.
                     // For each keypoint, only the center point will be
                     // drawn (without a circle around the keypoint with the
                     // keypoint size and orientation).
        DRAW_OVER_OUTIMG = 1, // Output image matrix will not be
                       // created (using Mat::create). Matches will be drawn
                       // on existing content of output image.
        NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
        DRAW_RICH_KEYPOINTS = 4 // For each keypoint, the circle around
                       // keypoint with keypoint size and orientation will
                       // be drawn.
    };
};
'''
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)
plt.imshow(img3)
plt.show()



