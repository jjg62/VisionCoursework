
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from sklearn.cluster import KMeans
from feature import Feature

#Des is a 2D array - an array of 128-dimensional feature vectors
#e.g. des[0] is the descriptor for the first feature


kpAngleAverage = []
allDes = np.empty((0, 128))
allkp = []

imageFeatureCounts = []
sift = cv.SIFT_create(sigma=1.6, contrastThreshold=0.04, nOctaveLayers=3)
imageNames = listdir('png')

for imageName in imageNames:
    img = cv.imread('png/' + imageName)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)

    allDes = np.append(allDes, des, axis=0)
    allkp.append(kp)
    imageFeatureCounts.append(len(des))


#allDes is an array of all feature vectors (over all images)

#Find vocabulary by doing KMeans on the descriptors
#idk = KMeans(n_clusters=1000, random_state=0).fit_predict(allDes)


def getClosestMatches(kp, descriptor):
    distances = np.zeros((allDes.shape[0]))

    for i in range(len(allDes)):
        distances[i] = np.linalg.norm(descriptor - allDes[i])

    features = []

    smallest = np.argmin(distances)
    smallestDist = distances[smallest]
    features.append(smallest)

    distances[smallest] = 999999999999
    secSmallest = np.argmin(distances)
    secSmallestDist = distances[secSmallest]

    #Ratio test: check that closest was much better than second closest
    if smallestDist < secSmallestDist * 0.85:

        #return image of smallest
        match = -1
        currentIdx = smallest
        while currentIdx >= 0:
            match += 1
            currentIdx -= imageFeatureCounts[match]

        feat = Feature(smallest, match, kp.pt, kp.angle, kp.size)
        return feat


    else:
        return None


testImg = cv.imread('test1/test_image_custom_1.png')
testImgGray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)
testKP, testDes = sift.detectAndCompute(testImgGray, None)

#plt.imshow(cv.drawKeypoints(testImgGray, testKP, testImg, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
#plt.show()

testVotes = np.zeros((len(imageNames)))

featuresFound = []
for i in range(len(imageNames)): featuresFound.append([])

anglesFound = []
for i in range(len(imageNames)): anglesFound.append([])

sizesFound = []
for i in range(len(imageNames)): sizesFound.append([])


for i in range(len(testDes)):
    k = testKP[i]
    d = testDes[i]
    feat = getClosestMatches(k, d)
    if(feat != None):
        # store the diff between keypoint angle of this and of kp
        # feat has an angle in it, need to find


        testVotes[feat.imageIdx] += 1
        featuresFound[feat.imageIdx].append(feat.pos)

        anglesFound[feat.imageIdx].append((allkp[feat.imageIdx]).angle - feat.angle)
        sizesFound[feat.imageIdx].append(allkp[feat.imageIdx].size - feat.size)


featuresNeededThreshold = 2

print(featuresFound)
for i in np.array(np.where(testVotes > featuresNeededThreshold))[0]:
    print(imageNames[i], testVotes[i])


boxedIm = testImg[:]

for i in range(len(featuresFound)):

    imFeatures = featuresFound[i]

    minX = 99999999999999999
    maxX = 0
    minY = 99999999999999999
    maxY = 0
    for featurePos in imFeatures:
        if(featurePos[0] > maxX):
            maxX = featurePos[0]

        if(featurePos[0] < minX):
            minX = featurePos[0]

        if(featurePos[1] > maxY):
            maxY = featurePos[1]

        if(featurePos[1] < minY):
            minY = featurePos[1]


    if len(imFeatures) > featuresNeededThreshold:
        boxedIm = cv.rectangle(boxedIm, (int(minX), int(minY)), (int(maxX), int(maxY)), (255, 0, 0), 4)
        cv.putText(boxedIm, imageNames[i], (int(minX), int(minY) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.imshow(boxedIm)
plt.show()