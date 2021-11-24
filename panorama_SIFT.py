import cv2
import numpy as np
import matplotlib.pyplot as plt



bloc1 = cv2.imread("1.png")
bloc2 = cv2.imread("2.png")
bloc3 = cv2.imread("3.png")
bloc1_g = cv2.cvtColor(bloc1,cv2.COLOR_BGR2GRAY)
bloc2_g = cv2.cvtColor(bloc2,cv2.COLOR_BGR2GRAY)
bloc3_g = cv2.cvtColor(bloc3,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()

# detect keypoints and descriptors for each image
kp1, des1 = sift.detectAndCompute(bloc1_g,None)
kp2, des2 = sift.detectAndCompute(bloc2_g,None)
kp3, des3 = sift.detectAndCompute(bloc3_g,None)

# Match features.
# Hamming-distance works only for binary feature-types like ORB, FREAK , that's why I use L2 norm
matcher = cv2.DescriptorMatcher_create(cv2.DIST_L2)

# Match features for the middle and 3rd image
matches2 = matcher.match(des2, des3, None)
# Sort matches by score
matches2.sort(key=lambda x: x.distance)
# Remove not so good matches
numGoodMatches = int(len(matches2) * 0.15)
matches2 = matches2[:numGoodMatches]

between2 = cv2.drawMatches(bloc2,kp2,bloc3,kp3,matches2,None,flags=2)

plt.figure(1)
plt.imshow(between2[...,::-1])


points2 = np.zeros((len(matches2), 2), dtype=np.float32)
points3 = np.zeros((len(matches2), 2), dtype=np.float32)
# extract the location of the good matches
for i, match in enumerate(matches2):
    points2[i, :] = kp2[match.queryIdx].pt
    points3[i, :] = kp3[match.trainIdx].pt
# Find homography
h2, mask2 = cv2.findHomography(points3, points2, cv2.RANSAC)
im1Height, im1Width, channels = bloc1.shape
im2Height, im2Width, channels = bloc2.shape
im3Height, im3Width, channels = bloc3.shape

im2Aligned2 = cv2.warpPerspective(bloc3, h2,(im2Width + im3Width, im3Height))
im2Aligned2[0:im2Height,0:im2Width] = bloc2
plt.figure(2)
plt.imshow(im2Aligned2[...,::-1])

aligned_gray = cv2.cvtColor(im2Aligned2,cv2.COLOR_BGR2GRAY)
kp, des = sift.detectAndCompute(aligned_gray,None)

### SECOND MATCH

matches = matcher.match(des1, des, None)
# Sort matches by score
matches.sort(key=lambda x: x.distance)
# Remove not so good matches
numGoodMatches = int(len(matches) * 0.15)
matches = matches[:numGoodMatches]

between1 = cv2.drawMatches(bloc1,kp1,im2Aligned2,kp,matches,None,flags=2)

plt.figure(3)
plt.imshow(between1[...,::-1])

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
# extract the location of the good matches
for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp[match.trainIdx].pt
# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
# Use homography
imHeight, imWidth, channels = im2Aligned2.shape
im2Aligned = cv2.warpPerspective(im2Aligned2, h,(imWidth + im1Width, im2Height))
im2Aligned[0:im1Height,0:im1Width] = bloc1
# im2Aligned = im2Aligned[:,:696]

plt.figure(4)
plt.imshow(im2Aligned[...,::-1])
plt.show()



