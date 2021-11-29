import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
#TODO: Fa o clasa cu metodele astea
#TODO: foloseste altceva in loc de bruteforce matcher

class Panorama:

    def __init__(self,images,crop=False, whole_picture = False):
        self.images = images
        self.fix_img_size(self.images)
        self.crop = crop
        self.whole_picture = whole_picture

    def stich_all_images(self):
        images = self.images[::-1]
        img_stiched = self.stich(images[1], images[0])
        for i in range(2,len(images)):
            if self.whole_picture:
                self.crop = False
                images[i] = cv2.copyMakeBorder(images[i],0,100,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            img_stiched = self.stich(images[i],img_stiched)

        if self.crop == True:
            img_stiched = self.crop_black_margins(img_stiched)

        return  img_stiched


    def stich(self,img1,img2):
        """

        :param img1: left image
        :param img2: right image
        Cu ajutorul matricei de homografie mapam kp de pe a doua imagine cu kp comune din prima imagine
        si aplicam o transformare de perspectiva asupra celei de a doua imagini
        peste rezultat lipim prima imagine pentru finaliza lipirea imaginilor
        :return:
        """
        img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        # descriptor - is a point of N-dimensional space.
        # match - is a pair of descriptors - one from the first set and one from the second set (also called train and query sets).
        # distance - is a L2 metric for 2 descriptors pointed by the match structure. (We are specifying the type of metric as a template parameter for BruteForceMatcher).

        # detect keypoints and descriptors for each image
        kp1,des1 = orb.detectAndCompute(img1_g,None)
        kp2,des2 = orb.detectAndCompute(img2_g,None)

        # VARIANTA BRUTE FORCE
        # match features - folosim hamming pentru ca descriptorii sunt stringuri de biti
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        #math features between the images
        matches = matcher.match(des1,des2,None)
        #sort matches by score
        matches.sort(key=lambda x: x.distance)

        # remove not so good matches
        numGoodMatches = int(len(matches)*0.15)
        matches = matches[:numGoodMatches]
        
        # varianta cu flann
        # FLANN_INDEX_LSH = 6
        # index_params = dict(algorithm=FLANN_INDEX_LSH,
        #                     table_number=6,  # 12
        #                     key_size=12,  # 20
        #                     multi_probe_level=1)  # 2
        # search_params = dict(checks=50)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # # Instead of returning the single best match for a given feature, KNN returns the k best matches.
        # all_matches = flann.knnMatch(des1, des2, k=2)  # returns 2 best matches for a given feature
        # # sanity check to view matches between images
        # matches = []
        # for m, n in all_matches:
        #     # For each pair of features (f1, f2), if the distance between f1 and f2 is within a certain ratio, we keep it
        #     # ideea e ca distanta sa fie suficient de mare, altfel matchul e ambiguu
        #     if m.distance / n.distance < 0.5:
        #         matches.append(m)


        between = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
        plt.figure(); plt.imshow(between[...,::-1])

        # extract the location of the good matches (x,y coordinates)
        # train is the image we learned (extracted features) beforehand,
        # query is the image that we are trying to match with the one trained.
        # trainIdx and queryIdx refer to the index of a point in the reference / query set respectively
        points1 = np.zeros((len(matches),2), dtype=np.float32)
        points2 = np.zeros((len(matches),2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i,:] = kp1[match.queryIdx].pt
            points2[i,:] = kp2[match.trainIdx].pt

        # The Homography is a 2D transformation. It maps points from one plane (image) to another.
        # Homography is very sensitive to the quality of data we pass to it.
        # Hence, it is important to have an algorithm (RANSAC) that can filter
        # points that clearly belong to the data distribution from the ones which do not.

        h, mask = cv2.findHomography(points2,points1,cv2.RANSAC,5.0)
        im1_h, im1_w, _ = img1.shape
        im2_h, im2_w, _ = img2.shape

        # perspective transform may combine one or more operations like rotation, scale, translation, or shear.
        # The idea is to transform one of the images so that both images merge as one
        # warp image2 to a common plane
        if self.whole_picture:
            im1Aligned = cv2.warpPerspective(img2, h, (im1_w + im2_w, im2_h+200))
        else:
            im1Aligned = cv2.warpPerspective(img2,h,(im1_w + im2_w,im2_h))
        plt.figure();plt.imshow(im1Aligned[..., ::-1])
        im1Aligned[0:im1_h,0:im1_w] = img1
        plt.figure();
        plt.imshow(im1Aligned[..., ::-1])
        return im1Aligned

    def equalize_histograms(self,images):
        ## TBD
        ref = images[2]
        multi = True if ref.shape[-1] > 1 else False
        for i in range(len(images)):
            images[i] = exposure.match_histograms(images[i],ref, multichannel=multi)

    def crop_black_margins(self,img):

        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        crop_col = []
        crop_linii_jos = []
        crop_linii_sus = []
        y_crop_jos = img_gray.shape[0]
        y_crop_sus = 0
        x_crop = img_gray.shape[1]

        # cautam linie cu linie pentru a afla indexul coloanei de unde incepe marginea neagra si trebuie sa taiem
        for i in range(img_gray.shape[0]):
            linie = img_gray[i, :]
            for j in range(len(linie)):
                if linie[j] == 0 and np.all(linie[j:] == 0) == True: # verific daca ce urmeaza e numai 0
                    crop_col.append(j) # salvam indicele coloanei unde incepe marginea neagra
        if len(crop_col) > 0:
            x_crop = min(crop_col)

        # cautam coloana cu coloana pentru a afla randul de unde trebuie sa taiem
        # DAR mergem PANA IN margine
        # altfel am avea coloane care incep direct cu 0
        for i in range(x_crop):
            col = img_gray[:, i]
            for k in range(len(col)):
                if col[k] == 0 and np.all(col[k:] == 0) == True:
                    crop_linii_jos.append(k)

            for k in reversed(range(len(col))):
                if col[k] == 0 and np.all(col[:k] == 0) == True:
                    crop_linii_sus.append(k)


        if len(crop_linii_jos) > 0:
            y_crop_jos = min(crop_linii_jos)

        if len(crop_linii_sus) > 0:
            y_crop_sus = min(crop_linii_jos)

        # crop imagine
        img = img[y_crop_sus:y_crop_jos, :x_crop, :]

        return img

    def fix_img_size(self,images):
        height_width = []
        for img in images:
            height_width.append([img.shape[0], img.shape[1]])
        h_min = min(height_width, key=lambda x: x[0])[0]
        w_min = min(height_width, key=lambda x: x[1])[1]
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (h_min, w_min))

if __name__ == '__main__':
    im1 = cv2.imread('parc1.jpg')
    im2 = cv2.imread('parc2.jpg')
    im3 = cv2.imread('parc3.jpg')
    images = [im1,im2,im3]
    panorama = Panorama(images,crop=True)
    stiched_image = panorama.stich_all_images()
    plt.figure()
    plt.imshow(stiched_image[..., ::-1]); plt.title("Stiched image")
    plt.show()
