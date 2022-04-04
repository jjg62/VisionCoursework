from os import listdir

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from match import Match

smallestTemplateSize = 16
largestTemplateSize = 512
templateShrinkRatio = 2


numberOfSizes = 8
numberOfRotations = 8


gaussianKSize = 5
gaussianStd = 1
pixelInterval = 1

correlationThreshold = 0.625

#For now, default image size (512x512) is the largest template size

def normalisedCrossCorrelation(normalisedTemplate, test, testX, testY):
    templateDimensions = normalisedTemplate.shape

    # Get subimage of test
    testSub = test[testY:testY + templateDimensions[1], testX:(testX + templateDimensions[0])]

    # Normalise Subimage
    normalisedTestSub = (testSub - testSub.mean()) / testSub.std()

    # Check correlation
    correlation = np.dot(normalisedTestSub.flatten(), normalisedTemplate.flatten()) / (
                templateDimensions[0] * templateDimensions[1])

    return correlation

def convolveNCC(template, test, instanceName):

    testDimensions = (test.shape[1], test.shape[0])
    templateDimensions = (template.shape[1], template.shape[0])

    normalisedTemplate = (template - template.mean()) / template.std()

    maxCorrelation = -2
    bestMatch = None

    for y in range(0, testDimensions[1] - templateDimensions[1], pixelInterval):
        for x in range(0, testDimensions[0] - templateDimensions[0], pixelInterval):

            correlation = normalisedCrossCorrelation(normalisedTemplate, test, x, y)

            if correlation > maxCorrelation:
                bestMatch = Match((x, y), templateDimensions, instanceName, correlation)
                maxCorrelation = correlation

    return bestMatch


imageNames = listdir('training')

allGoodMatches = []

#Read test img
testImg = cv.cvtColor(cv.imread('test_image_17.png'), cv.COLOR_BGR2GRAY)
#testImg[np.where(testImg > 250)] = 0 #Set background to black

multiResolutionRatio = 2
multiResolutionLayers = 3
multiResolutionThresholds = [0.85, 0.75, 0.7]

#Create arrays with images at each resolution
#Starting with biggest (actual size), shrinking by 2 each time
testImageMultiResolution = []

for layer in range(multiResolutionLayers):
    testImageMultiResolution.append(testImg)

    testImg = cv2.GaussianBlur(testImg, (gaussianKSize, gaussianKSize), gaussianStd)

    #Divide size by multiResolutionRation
    nextSize = (testImg.shape[0] // multiResolutionRatio, testImg.shape[1] // multiResolutionRatio)
    testImg = cv2.resize(testImg, nextSize)


def findBestMatch(layer, topLeft, bottomRight, size):
    global testImageMultiResolution, templateImageMultiResolution, multiResolutionThresholds, templateShrinkRatio

    # Get test and template image for this resolution
    test = testImageMultiResolution[layer]
    template = templateImageMultiResolution[layer]
    threshold = multiResolutionThresholds[layer]

    #plt.subplot(131), plt.imshow(test)

    #Crop test image to area we know template resides from previous call
    test = test[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]

    #plt.subplot(132), plt.imshow(test)

    currentSize = largestTemplateSize

    bestMatch = None
    bestCorrelation = -2

    while currentSize >= smallestTemplateSize:

        #If not on the smallest resolution (we know the size), skip every size except the one we want

        if size is None or size == currentSize:

            bestMatchAtThisSize = convolveNCC(template, test, imageName)

            if bestMatchAtThisSize is not None and bestMatchAtThisSize.correlation > bestCorrelation:
                bestMatch = bestMatchAtThisSize
                bestCorrelation = bestMatch.correlation

        currentSize = currentSize // templateShrinkRatio

        template = cv2.GaussianBlur(template, (gaussianKSize, gaussianKSize), gaussianStd)
        template = cv2.resize(template, (currentSize, currentSize))

    if bestMatch.correlation > threshold:
        if layer == 0:
            #Match was found on a sub-image, need to get co-ordinates on whole image
            bestMatch.pos = (bestMatch.pos[0] + topLeft[0], bestMatch.pos[1] + topLeft[1])
            return bestMatch
        else:
            newTopLeft = ((topLeft[0] + bestMatch.pos[0]-1)*multiResolutionRatio, (topLeft[1] + bestMatch.pos[1]-1)*multiResolutionRatio)
            newBottomRight = (newTopLeft[0] + (bestMatch.size[0] + 2)*multiResolutionRatio, newTopLeft[1] + (bestMatch.size[1] + 2)*multiResolutionRatio)

            return findBestMatch(layer-1, newTopLeft, newBottomRight, bestMatch.size[0]*multiResolutionRatio)

    #print("FAILED TO GET MATCH AT LAYER", layer, "CORRELATION:", bestMatch.correlation)
    return None



# Go to higher resolution


for imageName in imageNames:

    print(imageName)
    #Read image file and convert
    temp = cv.imread('training/' + imageName)
    templateGray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

    #Remove white background
    #templateGray[np.where(templateGray > 250)] = 0

    #Generate array of different resolutions of the template image
    templateImageMultiResolution = []

    for layer in range(multiResolutionLayers):
        templateImageMultiResolution.append(templateGray)

        templateGray = cv2.GaussianBlur(templateGray, (gaussianKSize, gaussianKSize), gaussianStd)

        # Divide size by multiResolutionRation
        nextSize = (templateGray.shape[0] // multiResolutionRatio, templateGray.shape[1] // multiResolutionRatio)
        templateGray = cv2.resize(templateGray, nextSize)

    smallestResolutionSize = ((testImageMultiResolution[multiResolutionLayers - 1]).shape[1], (testImageMultiResolution[multiResolutionLayers - 1]).shape[0])
    bestMatch = findBestMatch(multiResolutionLayers - 1, (0,0), smallestResolutionSize, None)
    if bestMatch is not None:
        allGoodMatches.append(bestMatch)
        print(len(allGoodMatches))

#Get highest resolution back
testImg = cv.cvtColor(testImageMultiResolution[0], cv.COLOR_GRAY2BGR)

#Draw bounding boxes
for match in allGoodMatches:
    testImg = cv.rectangle(testImg, match.pos, (match.pos[0] + match.size[0], match.pos[1] + match.size[1]), (255, 0, 0), 4)
    cv.putText(testImg, match.instanceName, (match.pos[0], match.pos[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.imshow(testImg)
plt.show()




