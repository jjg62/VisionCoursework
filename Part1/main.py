import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt

#Load in and convert image
pic = cv2.cvtColor(cv2.imread('ducks.jpeg'), cv2.COLOR_BGR2GRAY)
small_pic = pic[1200:1800, 1200:1800]

plt.imshow(cv2.cvtColor(small_pic, cv2.COLOR_GRAY2RGB))
plt.show()