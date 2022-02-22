import math

import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt



#Load in and convert image
pic = cv2.cvtColor(cv2.imread('angle/image3.png'), cv2.COLOR_BGR2GRAY)

plt.subplot(141), plt.imshow(cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)), plt.title('Original')

#Blur image to reduce effect of noise
#blurPic = cv2.GaussianBlur(pic, (9,9), 2)

#Find position of edge pixels
edges = cv2.Canny(pic, threshold1=100, threshold2=0)


plt.subplot(142), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), plt.title('Canny Edge Detection')


edgePixelLocations = np.transpose(np.array(np.where(edges > 0)))

imgDiagSize = math.sqrt(pic.shape[0]**2 + pic.shape[1]**2)

VOTE_THRESHOLD = 2

accResolutionRho = 450
accResolutionTheta = 90
accumulator = np.zeros((accResolutionTheta, accResolutionRho))

bestLine1 = (0, 0)
bestLine2 = (0, 0)


def testLines():
    global bestLine1, bestLine2
    for point in edgePixelLocations:
        #Iterate over angles from 0 to pi
        for i in range(accResolutionTheta):
            theta = i * math.pi / accResolutionTheta

            #Use point and theta to calculate rho
            rho = point[1] * math.cos(theta) + point[0] * math.sin(theta)

            #Get index of this rho value in the accumulator
            j = accResolutionRho * (rho + imgDiagSize) / (2 * imgDiagSize)
            j = round(j)

            accumulator[i, j] += 1

    plt.subplot(143), plt.imshow(accumulator), plt.title('Rho-Theta Hough Space'), plt.xlabel("Rho Index"), plt.ylabel("Theta Index")
    bestLine1 = np.unravel_index(accumulator.argmax(), accumulator.shape)

    #Eliminate that line and its neighbours within n?
    n = 5
    for k in range(n):
        for m in range(n):
            accumulator[bestLine1[0] - n//2 + k, bestLine1[1] - n//2 + m] = 0

    bestLine2 = np.unravel_index(accumulator.argmax(), accumulator.shape)


def drawLine(img, theta, rho):
    # If line is vertical/horizontal, exception cases

    if theta == math.pi/2:
        x1 = 0
        y1 = rho
        x2 = pic.shape[1]-1
        y2 = rho
    elif theta == 0:
        x1 = rho
        y1 = 0
        x2 = rho
        y2 = pic.shape[0]-1
    else:

        #First, try finding the point with x=0
        x1 = 0
        y1 = (rho - x1 * math.cos(theta)) / math.sin(theta)
        if y1 < 0:
            #If doesnt fit, set y1 to 0 instead and find x1
            y1 = 0
            x1 = (rho - y1 * math.sin(theta)) / math.cos(theta)
        elif y1 >= pic.shape[0]:
            y1 = pic.shape[0]
            x1 = (rho - y1 * math.sin(theta)) / math.cos(theta)

        #Same for a second point
        x2 = pic.shape[1]-1
        y2 = (rho - x2 * math.cos(theta)) / math.sin(theta)
        if y2 < 0:
            # If doesnt fit, set y1 to 0 instead and find x1
            y2 = 0
            x2 = (rho - y2 * math.sin(theta)) / math.cos(theta)
        elif y2 >= pic.shape[1]:
            y2 = pic.shape[0]-1
            x2 = (rho - y2 * math.sin(theta)) / math.cos(theta)

    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)

testLines()

print(bestLine1)
print(bestLine2)

img = edges[:]

drawLine(img, bestLine1[0] * math.pi / accResolutionTheta, (bestLine1[1] - accResolutionRho // 2) * 2 * imgDiagSize / accResolutionRho)
drawLine(img, bestLine2[0] * math.pi / accResolutionTheta, (bestLine2[1] - accResolutionRho // 2) * 2 * imgDiagSize / accResolutionRho)


plt.subplot(144), plt.imshow(img), plt.title('Redrawn Lines')

plt.show()






