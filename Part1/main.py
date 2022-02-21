import math

import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt



#Load in and convert image
pic = cv2.cvtColor(cv2.imread('angle/image10.png'), cv2.COLOR_BGR2GRAY)

#Blur image to reduce effect of noise
#blurPic = cv2.GaussianBlur(pic, (9,9), 2)

#Find position of edge pixels
edges = cv2.Canny(pic, threshold1=70, threshold2=110)

#Fit line
edgePixelLocations = np.transpose(np.array(np.where(edges != 0)))


imgDiagSize = math.sqrt(pic.shape[0]**2 + pic.shape[1]**2)

VOTE_THRESHOLD = 5

accResolution = 180
accumulator = np.zeros((accResolution, accResolution))


max1Votes = -1
bestLine1 = (0, 0)
max2Votes = -1
bestLine2 = (0, 0)
def updateMaxes(i, j):
    global max1Votes, bestLine1, max2Votes, bestLine2
    accumulator[i, j] += 1

    if i >= accResolution/2: return #Only consider max in the first 0-pi angles (change this later so we dont even look at them)

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
        for i in range(accResolution):
            theta = i * 2*math.pi/accResolution

            #Iterate over 90 line distances from 0 to imgDiagSize
            for j in range(accResolution):
                rho = (j-accResolution//2) * imgDiagSize/accResolution


                sqrDist = squareDistanceFromLine(theta, rho, point[0], point[1])
                #print(theta, sqrDist)

                if sqrDist < VOTE_THRESHOLD:
                    #Vote for that line
                    updateMaxes(i, j)

    plt.imshow(accumulator)
    plt.show()

    print(max1Votes)
    #print(np.transpose(np.array(np.where(accumulator > ))))

def squareDistanceFromLine(theta, rho, pointX, pointY):
    #If line is vertical/horizontal, exception cases
    if theta == math.pi/2 :
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


testLines()
angle1 = bestLine1[0] * 2*math.pi/accResolution
angle2 = bestLine2[0] * 2*math.pi/accResolution
print("RESULT: ", math.fabs(angle1-angle2))



