import math

import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt

#Load in and convert image
pic = cv2.cvtColor(cv2.imread('angle/image10.png'), cv2.COLOR_BGR2GRAY)

#Blur image to reduce effect of noise
blurPic = cv2.GaussianBlur(pic, (9,9), 2)

#Find position of edge pixels
edges = cv2.Canny(blurPic, threshold1=70, threshold2=110)

#Fit line
edgePixelLocations = np.transpose(np.array(np.where(edges != 0)))


imgDiagSize = math.sqrt(pic.shape[0]**2 + pic.shape[1]**2)

VOTE_THRESHOLD = 5

accResolution = 90
accumulator = np.zeros((accResolution, accResolution))




def testLines():

    # Keep track of (one for now) max
    maxVotes = -1
    bestLine = (0, 0)

    for point in edgePixelLocations:
        #Iterate over 90 line angles from 0 to 2pi
        for i in range(accResolution):
            theta = i * 2*math.pi/accResolution

            #Iterate over 90 line distances from 0 to imgDiagSize
            for j in range(90):
                rho = j * imgDiagSize/accResolution


                sqrDist = squareDistanceFromLine(theta, rho, point[0], point[1])
                #print(theta, sqrDist)

                if sqrDist < VOTE_THRESHOLD:
                    print(i)
                    #Vote for that line
                    accumulator[i, j] += 1

                    if(accumulator[i, j]) > maxVotes:
                        maxVotes = accumulator[i, j]
                        bestLine = (i, j)


    return bestLine


def squareDistanceFromLine(theta, rho, pointX, pointY):
    #If line is vertical, exception
    if theta == math.pi/2.0:
        return (pointX - rho)**2
    elif theta == 0:
        return (pointY - rho)**2
    else:
        #Get two random points on the line
        x1 = 1
        y1 = (rho - x1 * math.cos(theta)) / math.sin(theta)

        x2 = 10
        y2 = (rho - x2 * math.cos(theta)) / math.sin(theta)

        distSqr = (((x2 - x1)*(y1 - pointY) - (x1 - pointX)*(y2 - y1))**2)/((x2-x1)**2 + (y2-y1)**2)

        return distSqr


plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
plt.show()

print(testLines())


