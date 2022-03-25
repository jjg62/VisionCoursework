from os import listdir

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

smallestTemplateSize = 16
numberOfSizes = 8
numberOfRotations = 8
gaussianKSize = 7

#For now, defualt image size (512x512) is the largest template size


def normalisedCrossCorrelation(template, test):
    testDimensions = test.shape
    templateDimensions = template.shape

    normalisedTemplate = (template - template.mean()) / template.std()

    for y1 in range(0, testDimensions[1] - templateDimensions[1]):
        for x1 in range(0, testDimensions[0] - templateDimensions[0]):
            #Get subimage of test
            testSub = test[y1:y1+templateDimensions[1], x1:(x1+templateDimensions[0])]

            #Normalise Subimage
            normalisedTestSub = (testSub - testSub.mean()) // testSub.std()
            #print(testSub.std())

            #Check correlation
            correlation = np.dot(normalisedTestSub.flatten(), normalisedTemplate.flatten())
            #if correlation > 1200:
                #plt.imshow(cv.cvtColor(testSub, cv.COLOR_GRAY2RGB))
                #plt.show()


imageNames = listdir('training')
#for imageName in imageNames:

imageName = '011-trash.png'

#Read image file and convert
temp = cv.imread('training/' + imageName)
templateGray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

#Remove white background
templateGray[np.where(templateGray > 250)] = 0

#Scale it down - first apply Gaussian Blur
templateGray = cv2.GaussianBlur(templateGray, (gaussianKSize, gaussianKSize), 1)
templateGray = cv2.resize(templateGray, (50, 50))
#Later make it so many different sizes are used, for now just 1

#Slide/Convolve over image
testImg = cv.cvtColor(cv.imread('img.png'), cv.COLOR_BGR2GRAY)
testImg[np.where(testImg > 250)] = 0
plt.imshow(testImg)
plt.show()

normalisedCrossCorrelation(templateGray, testImg)







