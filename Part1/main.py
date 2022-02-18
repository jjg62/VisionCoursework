import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt

#Load in and convert image
pic = cv2.cvtColor(cv2.imread('angle/image1.png'), cv2.COLOR_BGR2GRAY)

#Blur image to reduce effect of noise
blurPic = cv2.GaussianBlur(pic, (9,9), 2)

#Find position of edge pixels
edges = cv2.Canny(blurPic, threshold1=70, threshold2=110)

#Fit line
edgePixelLocations = np.transpose(np.array(np.where(edges != 0)))

print(edgePixelLocations)
for point in edgePixelLocations:
    #go through every line
    #test distance
    #vote for one(s?) with smallest distance


#def distanceFromLine(m, c, )

#plt.imshow(cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB))
#plt.show()
