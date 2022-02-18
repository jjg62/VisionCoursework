import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt

#Load in and convert image
pic = cv2.cvtColor(cv2.imread('angle/image1.png'), cv2.COLOR_BGR2GRAY)


#Blur image to reduce effect of noise
pic = cv2.GaussianBlur(pic, (9,9), 2)

#Find position of edge pixels
pic = cv2.Canny(pic, threshold1=70, threshold2=110)

#Fit line

plt.imshow(cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB))
plt.show()
