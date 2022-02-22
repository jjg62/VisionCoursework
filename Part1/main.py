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
#Fit line
edgePixelLocations = np.transpose(np.array(np.where(edges > 0)))

imgDiagSize = math.sqrt(pic.shape[0]**2 + pic.shape[1]**2)

VOTE_THRESHOLD = 2

accResolutionX = 480
accResolutionY = 90
accumulator = np.zeros((accResolutionY, accResolutionX))


max1Votes = -1
bestLine1 = (0, 0)
max2Votes = -1
bestLine2 = (0, 0)
def updateMaxes(i, j):
    global max1Votes, bestLine1, max2Votes, bestLine2
    accumulator[i, j] += 1

    if(accumulator[i, j]) > max1Votes:
        # Move previous best to second best
        if bestLine1 != (i, j):
            max2Votes = max1Votes
            bestLine2 = bestLine1[:]
        # Update best
        max1Votes = accumulator[i, j]
        bestLine1 = (i, j)

    elif(accumulator[i, j]) > max2Votes:
        max2Votes = accumulator[i, j]
        bestLine2 = (i, j)


def testLines():
    for point in edgePixelLocations:
        #Iterate over 90 line angles from 0 to 2pi
        for i in range(accResolutionY):
            theta = i * math.pi/accResolutionY

            #Iterate over 90 line distances from 0 to imgDiagSize
            for j in range(accResolutionX):
                rho = (j-accResolutionX//2) * 2*imgDiagSize/accResolutionX


                sqrDist = squareDistanceFromLine(theta, rho, point[1], point[0])

                if sqrDist < VOTE_THRESHOLD**2:
                    #Vote for that line
                    updateMaxes(i, j)

    plt.subplot(143), plt.imshow(accumulator), plt.title('Rho-Theta Hough Space'), plt.xlabel("Rho Index"), plt.ylabel("Theta Index")

def squareDistanceFromLine(theta, rho, pointX, pointY):
    #If line is vertical/horizontal, exception cases
    if theta == math.pi/2:
        return (pointY - rho)**2
    elif theta == 0:
        return (pointX - rho)**2
    elif theta == 3*math.pi/2:
        return (pointY + rho) ** 2
    elif theta == math.pi:
        return (pointX + rho)**2
    else:
        #Get two random points on the line
        x1 = 1
        y1 = (rho - x1 * math.cos(theta)) / math.sin(theta)

        x2 = 10
        y2 = (rho - x2 * math.cos(theta)) / math.sin(theta)

        distSqr = (((x2 - x1)*(y1 - pointY) - (x1 - pointX)*(y2 - y1))**2)/((x2-x1)**2 + (y2-y1)**2)

        return distSqr



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
    print("dist1: ", squareDistanceFromLine(theta, rho, x1, y1))
    print("dist2: ", squareDistanceFromLine(theta, rho, x2, y2))

testLines()

print(bestLine1)
print(bestLine2)

img = edges[:]

drawLine(img, bestLine1[0] * math.pi/accResolutionY, (bestLine1[1] - accResolutionX//2) * 2*imgDiagSize/accResolutionX)
drawLine(img, bestLine2[0] * math.pi/accResolutionY, (bestLine2[1] - accResolutionX//2) * 2*imgDiagSize/accResolutionX)


plt.subplot(144), plt.imshow(img), plt.title('Redrawn Lines')

plt.show()

print("total edges: ", len(edges))
print("max votes 1: ",max1Votes)
print("max votes 2: ",max1Votes)






