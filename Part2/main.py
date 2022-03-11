import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from sklearn.cluster import KMeans

#Des is a 2D array - an array of 128-dimensional feature vectors
#e.g. des[0] is the descriptor for the first feature



allDes = np.empty((0, 128))


imageNames = listdir('png')
for imageName in imageNames:
    img = cv.imread('png/' + imageName)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)
    kp, des = sift.compute(gray, kp)

    allDes = np.append(allDes, des, axis=0)


#allDes is an array of all feature vectors (over all images)

#Find vocabulary by doing KMeans on the descriptors
idk = KMeans(n_clusters=1000, random_state=0).fit_predict(allDes)

print(idk)

