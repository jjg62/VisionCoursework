import math
from os import listdir

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from match import Match

#Parameters for multi-res analysis
multiResolutionRatio = 2 #Ratio to downsize each layer
multiResolutionLayers = 3 #Amount of layers (including original res)

#Thresholds for accepting matches at each resolution
#RECOMMENDED: [0.92, 0.8, 0.7] for fixed size
# [0.82, 0.75, 0.7] for variable size
multiResolutionThresholds = [0.92, 0.8, 0.7]

#Gaussian used to shrink test image
testGaussianKSize = 5
testGaussianStd = multiResolutionRatio / math.pi

#Gaussian Pyramid Parameters
smallestTemplateSize = 32 #RECOMMENDED: Use 16 for for variable size, 32 for fixed
largestTemplateSize = 512
#Ratio by which to shrink template each time (RECOMMENDED: Use 1.1 for variable size, 2 for fixed size)
templateShrinkRatio = 2

#Gaussian used to shrink templates
templateGaussianKSize = 5
templateGaussianStd = templateShrinkRatio / math.pi

#Stride for finding highest ncc
pixelStride = 1

#Names of training set
imageNames = listdir('training')

#For now, default image size (512x512) is the largest template size

#Get NCC between a template and a section on a test image with top-left position (testX, testY)
def normalisedCrossCorrelation(normalisedTemplate, test, testX, testY):
    templateDimensions = normalisedTemplate.shape

    # Get subimage of test
    testSub = test[testY:testY + templateDimensions[1], testX:(testX + templateDimensions[0])]

    std = testSub.std()
    if np.isnan(std) or std == 0: return -1 #Std of 0 means a blank area, not a match

    # Normalise Subimage
    normalisedTestSub = (testSub - testSub.mean()) / std

    # Check correlation
    correlation = np.dot(normalisedTestSub.flatten(), normalisedTemplate.flatten()) / (
                templateDimensions[0] * templateDimensions[1])

    return correlation

#Find NCC at every position, return the highest for a given template
def slideNCC(template, test, instanceName):

    testDimensions = (test.shape[1], test.shape[0])
    templateDimensions = (template.shape[1], template.shape[0])

    #Normalise template - subtract mean and divide by std
    normalisedTemplate = (template - template.mean()) / template.std()

    #Keep track of best match
    maxCorrelation = -2
    bestMatch = None

    #For each position in the image
    for y in range(0, testDimensions[1] - templateDimensions[1], pixelStride):
        for x in range(0, testDimensions[0] - templateDimensions[0], pixelStride):

            #Check NCC
            correlation = normalisedCrossCorrelation(normalisedTemplate, test, x, y)

            #Check if it's the new best
            if correlation > maxCorrelation:
                bestMatch = Match((x, y), templateDimensions, instanceName, correlation)
                maxCorrelation = correlation


    return bestMatch


#Find best match in a given layer of the multi-res analysis, recursively search higher res test images if found
def findBestMatch(layer, topLeft, bottomRight, size, imageName):
    global testImageMultiResolution, templateImageMultiResolution, multiResolutionThresholds, templateShrinkRatio

    # Get test and template image for this resolution
    test = testImageMultiResolution[layer]
    template = templateImageMultiResolution[layer]
    threshold = multiResolutionThresholds[layer]


    #Crop test image to area we know template resides from previous call
    test = test[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]

    #Find largest template size for this resolution
    currentSize = largestTemplateSize // multiResolutionRatio ** layer

    #Keep track of best match
    bestMatch = None
    bestCorrelation = -2

    #While template size is more than smallest for this resolution
    while currentSize >= smallestTemplateSize // multiResolutionRatio ** layer:

        #If not on the smallest resolution (i.e. when we know the size that should be checked), skip every size except ones near the desired size
        if size is None or abs(size - currentSize) < 6:
            bestMatchAtThisSize = slideNCC(template, test, imageName) #Find best match for this template size

            #If a match was found which is better than current best, it becomes the new best
            if bestMatchAtThisSize is not None and bestMatchAtThisSize.correlation > bestCorrelation:
                bestMatch = bestMatchAtThisSize
                bestCorrelation = bestMatch.correlation

        #Decrease template size
        currentSize = int(currentSize // templateShrinkRatio)

        if templateShrinkRatio < 2:
            #Use cubic interpolation for small shrink ratio
            template = cv2.resize(template, (currentSize, currentSize), interpolation=cv.INTER_CUBIC)
        else:
            #Otherwise use standard Gaussian Pyramid method
            template = cv2.GaussianBlur(template, (templateGaussianKSize, templateGaussianKSize), templateGaussianStd)
            template = cv2.resize(template, (currentSize, currentSize), interpolation=cv.INTER_NEAREST)

    #If a match was found that passes the threshold
    if bestMatch is not None and bestMatch.correlation > threshold:
        if layer == 0: #Match made on original resolution
            #Match was found on a sub-image, need to get co-ordinates on whole image
            bestMatch.pos = (bestMatch.pos[0] + topLeft[0], bestMatch.pos[1] + topLeft[1])
            return bestMatch
        else:
            #Find a suitable neighbouhood around the area the template was found to search in the next highest resolution
            newTopLeft = ((topLeft[0] + bestMatch.pos[0]-1)*multiResolutionRatio, (topLeft[1] + bestMatch.pos[1]-1)*multiResolutionRatio)
            newBottomRight = (newTopLeft[0] + (bestMatch.size[0] + 2)*multiResolutionRatio, newTopLeft[1] + (bestMatch.size[1] + 2)*multiResolutionRatio)

            #Recursively search next highest res
            return findBestMatch(layer-1, newTopLeft, newBottomRight, bestMatch.size[0]*multiResolutionRatio, imageName)

    #Match did not pass threshold
    return None

#Store test and templates at each resolution in global variables
testImageMultiResolution = []
templateImageMultiResolution = []

def recognize(testImageName, showResult):

        global testImageMultiResolution, templateImageMultiResolution

        allGoodMatches = []

        #Read and convert input test img
        testImg = cv.cvtColor(cv.imread(testImageName), cv.COLOR_BGR2GRAY)
        #testImg[np.where(testImg > 230)] = 0 #Set background to black

        #Create arrays with images at each resolution
        #Starting with biggest (actual size), shrinking by 2 each time
        testImageMultiResolution = []
        for layer in range(multiResolutionLayers):
            testImageMultiResolution.append(testImg)

            #Blur image
            testImg = cv2.GaussianBlur(testImg, (testGaussianKSize, testGaussianKSize), testGaussianStd)

            #Divide size by multiResolutionRatio
            nextSize = (testImg.shape[0] // multiResolutionRatio, testImg.shape[1] // multiResolutionRatio)
            testImg = cv2.resize(testImg, nextSize, cv.INTER_NEAREST)


        #For each template image
        for imageName in imageNames:

            print("Trying " + imageName + "...")
            #Read image file and convert
            temp = cv.imread('training/' + imageName)
            templateGray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

            #Remove white background
            #templateGray[np.where(templateGray > 230)] = 0

            #Generate array of different resolutions of the template image
            templateImageMultiResolution = []
            for layer in range(multiResolutionLayers):
                templateImageMultiResolution.append(templateGray)

                #Blur Image
                templateGray = cv2.GaussianBlur(templateGray, (templateGaussianKSize, templateGaussianKSize), templateGaussianStd)

                # Divide size by multiResolutionRatio
                nextSize = (templateGray.shape[0] // multiResolutionRatio, templateGray.shape[1] // multiResolutionRatio)
                templateGray = cv2.resize(templateGray, nextSize, cv.INTER_NEAREST)

            #Get size of smallest resolution of test image - need this to tell findBestMatch to search whole image
            smallestResolutionSize = ((testImageMultiResolution[multiResolutionLayers - 1]).shape[1], (testImageMultiResolution[multiResolutionLayers - 1]).shape[0])

            #Call on highest layer (smallest resolution)
            bestMatch = findBestMatch(multiResolutionLayers - 1, (0,0), smallestResolutionSize, None, imageName)

            #Keep track of all templates which match
            if bestMatch is not None:
                allGoodMatches.append(bestMatch)


        if showResult:
            #Get highest resolution back
            testImg = cv.cvtColor(testImageMultiResolution[0], cv.COLOR_GRAY2BGR)

            #Draw bounding boxes
            for match in allGoodMatches:
                testImg = cv.rectangle(testImg, match.pos, (match.pos[0] + match.size[0], match.pos[1] + match.size[1]), (255, 0, 0), 4)
                cv.putText(testImg, match.instanceName, (match.pos[0], match.pos[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            plt.imshow(testImg)
            plt.show()

        return allGoodMatches


#--USE THIS TO TEST ON ONE IMAGE--
#Otherwise, run test.py
inputName = "test/test_image_1.png"
#Uncomment line below then run
#recognize(inputName, True)

#(keep it commented out if using test.py)
