import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import exposure
class Panorama:

    def __init__(self,images,crop=False, whole_picture = False):
        self.images = images
        self.fix_img_size(self.images)
        # self.equalize_histograms(self.images)
        self.crop = crop
        self.whole_picture = whole_picture

    def stitch_all_images(self):
        # inversam lista de imagini pentru ca vrem sa incepem lipirea imaginilor de la ultima la prima
        # deoarece cand facem lipirea penultima imagine ramane neschimbata in timp ce ultimei imagini i se va modifica
        # perspectiva dupa penultima si va fi lipita peste ea
        # apoi imaginea lipita va lua rolul ultimei imagini urmand sa fie modificata pentru urmatoarea imagine si tot asa
        images = self.images[::-1]
        img_stiched = self.stich(images[1], images[0])
        for i in range(2,len(images)):
            # ma asigur ca flagul de crop este False daca vrem sa vedem toata
            if self.whole_picture:
                self.crop = False

            img_stiched = self.stich(images[i],img_stiched)

        # daca flag-ul pentru crop este True atunci apelam metoda de eliminare a marginilor negre
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
        :return imagini lipite:
        """
        ##TODO: comparatie algoritmi cu time sa vedem cat de repede merg
        # facem imaginile grayscale
        img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        # creare obiect pentru ORB feature descriptor
        orb = cv2.ORB_create()
        # descriptor - un vector de N dimensiuni
        # match - o pereche de descriptori -unul din primul set, unul din al doilea set (numite si train and query sets).

        # detecteaza keypoints and descriptori pentru fiecare imagine
        kp1,des1 = orb.detectAndCompute(img1_g,None)
        kp2,des2 = orb.detectAndCompute(img2_g,None)

        # VARIANTA BRUTE FORCE
        # match features - folosim hamming pentru ca descriptorii sunt stringuri de biti
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        # #math features between the images
        matches = matcher.match(des1,des2,None)
        #sort matches by score
        matches.sort(key=lambda x: x.distance)

        # pastram cele mai bune matches
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

        # afisare corespondenta intre keypoints
        between = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
        plt.figure(); plt.imshow(between[...,::-1])

        # extrage locatia pentru matches bune (x,y coordinates)
        # train e imaginea pe care am "invatat-o" (am extras features) inainte,
        # query este imaginea pe care incercam sa o potrivim cu imaginea invatata.
        # trainIdx si queryIdx se refera la indexul punctului din seteul de referinta / respectiv query
        points1 = np.zeros((len(matches),2), dtype=np.float32)
        points2 = np.zeros((len(matches),2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i,:] = kp1[match.queryIdx].pt
            points2[i,:] = kp2[match.trainIdx].pt

        # Homography este o transformare 2D ce mapeaza puncte dintr-un plan (imagine) in altul.
        # Este foarte sensibila la calitate punctelor ce se folosesc pentru calcularea ei
        # asadar, este important sa folosim un algoritm ca RANSAC care poate filtra punctele
        # ce apartin distributiei de date (inliers) de cei care nu fac parte (outliers)
        # points2 source points , points1 target points
        # 5 reprezinta ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
        h, mask = cv2.findHomography(points2,points1,cv2.RANSAC,5.0)
        im1_h, im1_w, _ = img1.shape
        im2_h, im2_w, _ = img2.shape

        # perspective transform may combine one or more operations like rotation, scale, translation, or shear.
        # The idea is to transform one of the images so that both images merge as one
        # warp image2 to a common plane
        # daca vrem sa vedem toata transformarea de perspectiva maresc dimensiunea imaginii finale cu o margine
        # de inca 300 px
        if self.whole_picture:
            im1Aligned = cv2.warpPerspective(img2, h, (im1_w + im2_w, im2_h+300))
        else:
            im1Aligned = cv2.warpPerspective(img2,h,(im1_w + im2_w,im2_h))
        plt.figure();plt.imshow(im1Aligned[..., ::-1])
        # lipirea imaginii nemodificata
        im1Aligned[0:im1_h,0:im1_w] = img1
        plt.figure();
        plt.imshow(im1Aligned[..., ::-1])
        return im1Aligned


    def crop_black_margins(self,img):

        # fac imaginea grayscale pentru simplitate, marginile fiind negre o sa aiba valoarea 0
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        crop_col = [] # lista pentru a salva indicii de pe coloane unde ar incepe marginea din dreapta
        crop_linii_jos = [] # lista pentru a salva indicii de pe linii unde ar incepe marginea de jos
        crop_linii_sus = [] # lista pentru a salva indicii de pe linii unde ar incepe marginea de sus

        # initializare cu valorile default pentru crop
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
            # gasim cel mai din stanga index
            x_crop = min(crop_col)

        # cautam coloana cu coloana pentru a afla randul de unde trebuie sa taiem
        # DAR mergem PANA IN margine
        # altfel am avea coloane care incep direct cu 0
        for i in range(x_crop):
            col = img_gray[:, i]
            # caut marginea de jos
            for k in range(len(col)):
                if col[k] == 0 and np.all(col[k:] == 0) == True:
                    crop_linii_jos.append(k)
            # caut marginea de sus
            for k in reversed(range(len(col))):
                if col[k] == 0 and np.all(col[:k] == 0) == True:
                    crop_linii_sus.append(k)


        if len(crop_linii_jos) > 0:
            y_crop_jos = min(crop_linii_jos) # vrem cea mai inalta linie

        if len(crop_linii_sus) > 0:
            y_crop_sus = max(crop_linii_sus) # vrem cea mai joasa linie


        # crop imagine
        img = img[y_crop_sus:y_crop_jos, :x_crop, :]

        return img

    def fix_img_size(self,images):
        # vrem ca toate imaginile sa aiba aceleasi dimensiuni
        # asa ca redimensionam toate imaginile dupa dimensiunile minime intalnite
        # astfel incat sa nu pierdem nici detalii
        height_width = []
        for img in images:
            height_width.append([img.shape[0], img.shape[1]])
        h_min = min(height_width, key=lambda x: x[0])[0]
        w_min = min(height_width, key=lambda x: x[1])[1]
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (w_min, h_min))

    def equalize_histograms(self, images):
        ## TBD
        ref = images[0]
        multi = True if ref.shape[-1] > 1 else False
        for i in range(len(images)):
            images[i] = exposure.match_histograms(images[i], ref, multichannel=multi)

if __name__ == '__main__':
    im1 = cv2.imread('deal1.jpg')
    im2 = cv2.imread('deal2.jpg')
    im3 = cv2.imread('deal3.jpg')
    # im4 = cv2.imread('iarna4.jpg')
    # lista cu imaginile in ordine
    images = [im1,im2,im3]
    panorama = Panorama(images,crop=True)
    start = time.time()
    stiched_image = panorama.stitch_all_images()
    print(f"Timp pentru procesare {time.time() - start}")
    plt.figure()
    plt.imshow(stiched_image[..., ::-1]); plt.title("Stiched image")
    plt.show()
