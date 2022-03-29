from os import listdir

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

smallestTemplateSize = 16
largestTemplateSize = 512
numberOfSizes = 8
numberOfRotations = 8
gaussianKSize = 7
pixelInterval = 4

correlationThreshold = 0.85

#For now, defualt image size (512x512) is the largest template size


def normalisedCrossCorrelation(template, test):

    testDimensions = test.shape
    templateDimensions = template.shape

    normalisedTemplate = (template - template.mean()) / template.std()

    maxCorrelation = -999999999
    maxPosition = (0, 0)

    for y1 in range(0, testDimensions[1] - templateDimensions[1], pixelInterval):
        for x1 in range(0, testDimensions[0] - templateDimensions[0], pixelInterval):
            #Get subimage of test
            testSub = test[y1:y1+templateDimensions[1], x1:(x1+templateDimensions[0])]

            #Normalise Subimage
            normalisedTestSub = (testSub - testSub.mean()) // testSub.std()
            #print(testSub.std())

            #Check correlation
            correlation = np.dot(normalisedTestSub.flatten(), normalisedTemplate.flatten()) / (templateDimensions[0] * templateDimensions[1])
            if(correlation > maxCorrelation):
                maxCorrelation = correlation
                maxPosition = (x1, y1)

    print(maxCorrelation)
    plt.imshow(test[maxPosition[1]:maxPosition[1] + templateDimensions[1],
               maxPosition[0]:maxPosition[0] + templateDimensions[0]])
    plt.show()

    return maxCorrelation



imageNames = listdir('training')
#for imageName in imageNames:

imageName = '016-house.png'

#Read image file and convert
temp = cv.imread('training/' + imageName)
templateGray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

#Remove white background
templateGray[np.where(templateGray > 250)] = 0

#Read test img
testImg = cv.cvtColor(cv.imread('img.png'), cv.COLOR_BGR2GRAY)
testImg[np.where(testImg > 250)] = 0 #Set background to black

plt.imshow(cv.cvtColor(templateGray, cv.COLOR_GRAY2RGB));
plt.show()

for i in range(numberOfSizes, -1, -1):

    sizeInterval = (largestTemplateSize - smallestTemplateSize) // numberOfSizes
    currentSize = smallestTemplateSize + sizeInterval * i

    print(currentSize)

    templateGray = cv2.GaussianBlur(templateGray, (gaussianKSize, gaussianKSize), 1)
    templateGray = cv2.resize(templateGray, (currentSize, currentSize))

    normalisedCrossCorrelation(templateGray, testImg)








