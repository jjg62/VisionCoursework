import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from feature import Feature

#Database of training image descriptors / keypoints
allDes = np.empty((0, 128))

#Initialise SIFT
sift = cv.SIFT_create(contrastThreshold=0.02, nOctaveLayers=9)
imageFeatureCounts = [] #Keep track of how many features are in each image, such that image label can be retrieved
imageNames = listdir('png')

#Get descriptors from training images
def train():
    global allDes, imageNames, imageFeatureCounts

    for imageName in imageNames:
        #Read and convert image
        img = cv.imread('png/' + imageName)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Get keypoints and descriptors
        kp, des = sift.detectAndCompute(gray, None)

        #Add descriptors to the to database
        allDes = np.append(allDes, des, axis=0)
        imageFeatureCounts.append(len(des))

#Find closest descriptor in database and apply ratio test
def getClosestMatches(kp, descriptor, featuresFound):

    #Find distance from each descriptor in database to this one
    distances = np.zeros((allDes.shape[0]))
    for i in range(len(allDes)):
        distances[i] = np.linalg.norm(descriptor - allDes[i])

    #Find smallest distance
    smallest = np.argmin(distances)
    smallestDist = distances[smallest]

    #Find second smallest distance
    distances[smallest] = 999999999999
    secSmallest = np.argmin(distances)
    secSmallestDist = distances[secSmallest]

    #Find the template image this feature belonged to
    match = -1
    currentIdx = smallest
    while currentIdx >= 0:
        match += 1
        currentIdx -= imageFeatureCounts[match]

    #Ratio test: check that closest was much better than second closest
    if smallestDist < secSmallestDist * 0.6:
        feat = Feature(smallest, match, kp.pt, kp.angle, kp.size)
        return feat

    elif featuresFound[match]:
        #If features have already been found with this classification, give an advantage if they're near
        for pos in featuresFound[match]:
            distSqr = (pos[0] - kp.pt[0])**2 + (pos[1] - kp.pt[1])**2
            if distSqr < 3000 and smallestDist < secSmallestDist * 0.9:
                feat = Feature(smallest, match, kp.pt, kp.angle, kp.size)
                return feat

    else:
        return None


#Test an input image
def test(testImageName, show):
    #Read and convert input image
    testImg = cv.imread(testImageName)
    testImgGray = cv.cvtColor(testImg, cv.COLOR_BGR2GRAY)

    #Get keypoints and descriptors from test image
    testKP, testDes = sift.detectAndCompute(testImgGray, None)

    #Number of matches found for each instance
    testVotes = np.zeros((len(imageNames)))

    #Store position of features found so far to check spatial consistency
    featuresFound = []
    for i in range(len(imageNames)): featuresFound.append([])


    #For each feature found
    for i in range(len(testDes)):
        k = testKP[i]
        d = testDes[i]
        feat = getClosestMatches(k, d, featuresFound)
        if(feat != None):
            testVotes[feat.imageIdx] += 1
            featuresFound[feat.imageIdx].append(feat.pos)


    outputMatchNames = []

    #Find names of all instances for which more than 4 features were found
    featuresNeededThreshold = 4
    for i in np.array(np.where(testVotes > featuresNeededThreshold))[0]:
        outputMatchNames.append(imageNames[i][4:-4])

    #Display results
    if(show):
        #Create copy of the original images
        boxedIm = testImg[:]

        for i in range(len(featuresFound)):

            imFeatures = featuresFound[i]

            #Find top-left and bottom-right for a box that surrounds all features for this instance
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


            #If the number of features pass the threshold, show the box
            if len(imFeatures) > featuresNeededThreshold:
                boxedIm = cv.rectangle(boxedIm, (int(minX), int(minY)), (int(maxX), int(maxY)), (255, 0, 0), 4)
                cv.putText(boxedIm, imageNames[i], (int(minX), int(minY) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.imshow(cv.cvtColor(boxedIm, cv2.COLOR_BGR2RGB))
        plt.show()

    return outputMatchNames

#--USE THIS TO TEST ON ONE IMAGE--
#Otherwise, run test.py
inputName = "test2/test_image_1.png"
#Uncomment lines below then run

train()
test(inputName, True)

#(keep it commented out if using test.py)
