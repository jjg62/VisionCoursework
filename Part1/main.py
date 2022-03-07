import math
import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt



#Load in and convert image
pic = cv2.cvtColor(cv2.imread('angle/image7.png'), cv2.COLOR_BGR2GRAY)

plt.subplot(221), plt.imshow(cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)), plt.title('Original')

#Blur image to reduce effect of noise
blurPic = cv2.GaussianBlur(pic, (9,9), 2)

#Find position of edge pixels
edges = cv2.Canny(pic, threshold1=100, threshold2=0)

#Plot Results of Edge Detection
plt.subplot(222), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), plt.title('Canny Edge Detection')

#Find co-ordinates of points detected
edgePixelLocations = np.transpose(np.array(np.where(edges > 0)))

#Get a maximum rho size by taking the diagonal size of image
imgDiagSize = math.sqrt(pic.shape[0]**2 + pic.shape[1]**2)

#Resolutions of accumulator
accResolutionRho = 1000
accResolutionTheta = 180
accumulator = np.zeros((accResolutionTheta, accResolutionRho))

bestLine1 = (0, 0)
bestLine2 = (0, 0)

#Find two best fitting lines from Hough Transform
def houghLines():
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

            #Add a vote to the accumulator
            accumulator[i, j] += 1

    #Plot the Hough space
    plt.subplot(223), plt.imshow(accumulator), plt.title('Rho-Theta Hough Space'), plt.xlabel("Rho Index"), plt.ylabel("Theta Index")

    #Get co-ordinates of best line
    bestLine1 = np.unravel_index(accumulator.argmax(), accumulator.shape)

    #Eliminate that line and its neighbouring n//2 rows of co-ordinate pairs (such that 2 similar angles are not picked)
    n = 9
    for k in range(n):
        for m in range(n):
            accumulator[bestLine1[0] - n//2 + k] = 0

    #Now get the maximum again for second line
    bestLine2 = np.unravel_index(accumulator.argmax(), accumulator.shape)


def findIntersection():
    global bestLine1, bestLine2

    #Convert accumulator coordinates to angles/distances
    theta1 = bestLine1[0] * math.pi / accResolutionTheta
    theta2 = bestLine2[0] * math.pi / accResolutionTheta

    rho1 = (bestLine1[1] * 2 * imgDiagSize / accResolutionRho) - imgDiagSize
    rho2 = (bestLine2[1] * 2 * imgDiagSize / accResolutionRho) - imgDiagSize

    #Check exceptions first - if each line is vertical/horizontal

    if theta1 == 0: #Angle is 0, vertical
        if theta2 == 0: return (-1, -1) #Error - shouldn't have parallel lines
        else:
            x = rho1
            y = (rho2 - x * math.cos(theta2)) / math.sin(theta2)

    elif theta1 == math.pi/2: #Line is horizontal
        if theta2 == math.pi/2: return(-1, -1)
        else:
            y = rho1
            x = (rho2 - y * math.sin(theta2)) / math.cos(theta2)

    else:
        #Check line 2
        if theta2 == 0:  # Angle is 0, vertical
            if theta1 == 0:
                return None  # Error - shouldn't have parallel lines
            else:
                x = rho2
                y = (rho1 - x * math.cos(theta1)) / math.sin(theta1)

        elif theta2 == math.pi / 2:  # Line is horizontal
            if theta1 == math.pi / 2:
                return None
            else:
                y = rho2
                x = (rho1 - y * math.sin(theta1)) / math.cos(theta1)

        else:
            #Neither of the lines are exceptions, can solve a simulatneous equation for intersection
            x = (rho2 * math.sin(theta1) - rho1 * math.sin(theta2)) / (math.cos(theta2) * math.sin(theta1) - math.cos(theta1) * math.sin(theta2))
            y = (rho1 - x * math.cos(theta1)) / math.sin(theta1)

    return (x, y)


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

        #Calculate distance using a vector made from the two points
        distSqr = (((x2 - x1)*(y1 - pointY) - (x1 - pointX)*(y2 - y1))**2)/((x2-x1)**2 + (y2-y1)**2)

        return distSqr

#We have the parameters for the two lines, but we don't know where to measure the angle
def ensureCorrectAngle(foundAngle):
    global bestLine1, bestLine2

    #Convert accumulator coordinates to angles/distances
    theta1 = bestLine1[0] * math.pi / accResolutionTheta
    theta2 = bestLine2[0] * math.pi / accResolutionTheta

    rho1 = (bestLine1[1] * 2 * imgDiagSize / accResolutionRho) - imgDiagSize
    rho2 = (bestLine2[1] * 2 * imgDiagSize / accResolutionRho) - imgDiagSize

    #Find intersection
    p0 = findIntersection()
    p1 = None
    p2 = None

    lineDistThreshold = 2
    maxDistLine1 = -1
    maxDistLine2 = -1

    #Find two points - one on the first line, one on the second - that are as far away from intersection as possible
    for edge in edgePixelLocations:
        dist = (edge[1] - p0[0]) ** 2 + (edge[0] - p0[1]) ** 2
        if squareDistanceFromLine(theta1, rho1, edge[1], edge[0]) < lineDistThreshold**2:
            if dist > maxDistLine1:
                p1 = (edge[1], edge[0])
                maxDistLine1 = dist
        elif squareDistanceFromLine(theta2, rho2, edge[1], edge[0]) < lineDistThreshold**2:
            if dist > maxDistLine2:
                p2 = (edge[1], edge[0])
                maxDistLine2 = dist

    #Output angle between the vectors P1-P0 and P2-P0
    angle1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    angle2 = math.atan2(p2[1] - p0[1], p2[0] - p0[0])
    between = math.fabs(angle2 - angle1) * 360 / (2 * math.pi)
    otherAngle = math.fabs(foundAngle - 180)

    #Which is closer to the angle found in this function: the original one or 180-it?
    if math.fabs(foundAngle - between) < math.fabs(otherAngle - between):
        return foundAngle
    else:
        return otherAngle


#Draw lines on a plot, from their rho and theta parameters
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

#Find 2 Best fitting lines, save them to bestLine1 and bestLine2
houghLines()

#Find angle between the two theta parameters
angle = math.fabs(bestLine1[0] - bestLine2[0]) * 180/accResolutionTheta

#Copy the Canny image to draw the lines on
img = edges[:]

#Draw the two found lines onto img
drawLine(img, bestLine1[0] * math.pi / accResolutionTheta, (bestLine1[1] - accResolutionRho // 2) * 2 * imgDiagSize / accResolutionRho)
drawLine(img, bestLine2[0] * math.pi / accResolutionTheta, (bestLine2[1] - accResolutionRho // 2) * 2 * imgDiagSize / accResolutionRho)

#Use ensureCorrectedAngle to make sure the angle is the BETWEEN the lines on the image, not 180 - it
print("ANGLE:", ensureCorrectAngle(angle))

#Show the drawn lines
plt.subplot(224), plt.imshow(img), plt.title('Redrawn Lines')

plt.show()






