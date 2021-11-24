import cv2
import numpy as np
import matplotlib.pyplot as plt

bloc1 = cv2.imread("1.png")
bloc2 = cv2.imread("2.png")
bloc3 = cv2.imread("3.png")
bloc1_g = cv2.cvtColor(bloc1,cv2.COLOR_BGR2GRAY)
bloc2_g = cv2.cvtColor(bloc2,cv2.COLOR_BGR2GRAY)
bloc3_g = cv2.cvtColor(bloc3,cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()

# descriptor - is a point of N-dimensional space.
# match - is a pair of descriptors - one from the first set and one from the second set (also called train and query sets).
# distance - is a L2 metric for 2 descriptors pointed by the match structure. (We are specifying the type of metric as a template parameter for BruteForceMatcher).

# detect keypoints and descriptors for each image
kp1, des1 = orb.detectAndCompute(bloc1_g,None)
kp2, des2 = orb.detectAndCompute(bloc2_g,None)
kp3, des3 = orb.detectAndCompute(bloc3_g,None)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

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
#train is the image we learned (extracted features) beforehand,
#query is the image that we are trying to match with the one trained.
#trainIdx and queryIdx refer to the index of a point in the reference / query set respectively
for i, match in enumerate(matches2):
    points2[i, :] = kp2[match.queryIdx].pt
    points3[i, :] = kp3[match.trainIdx].pt
#The Homography is a 2D transformation. It maps points from one plane (image) to another.
#Homography is very sensitive to the quality of data we pass to it.
# Hence, it is important to have an algorithm (RANSAC) that can filter
# points that clearly belong to the data distribution from the ones which do not.
# Find homography
h2, mask2 = cv2.findHomography(points3, points2, cv2.RANSAC)
im1Height, im1Width, channels = bloc1.shape
im2Height, im2Width, channels = bloc2.shape
im3Height, im3Width, channels = bloc3.shape

#perspective transform may combine one or more operations like rotation, scale, translation, or shear.
#The idea is to transform one of the images so that both images merge as one
# warp bloc3 image to a common plane
im2Aligned2 = cv2.warpPerspective(bloc3, h2,(im2Width + im3Width, im3Height))
im2Aligned2[0:im2Height,0:im2Width] = bloc2
plt.figure(2)
plt.imshow(im2Aligned2[...,::-1])

### SECOND Stitching
aligned_gray = cv2.cvtColor(im2Aligned2,cv2.COLOR_BGR2GRAY)
kp, des = orb.detectAndCompute(aligned_gray,None)

# We match the first image with the previously stitched image

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
# Crop the black part
im2Aligned_cropped = im2Aligned[:,:696]

plt.figure(4)
plt.imshow(im2Aligned[...,::-1]);plt.title("Before cropping")
plt.figure(5)
plt.imshow(im2Aligned_cropped[...,::-1]);plt.title("Final result")
plt.show()

