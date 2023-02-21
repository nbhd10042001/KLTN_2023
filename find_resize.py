import cv2 
import numpy as np
import matplotlib.pyplot as plt



def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)  
    return canny  

image = cv2.imread("img/lane/test_image.jpg")
image = cv2.resize(image, [1250, 700])
lane_image = np.copy(image)
canny_image = canny(lane_image)

plt.imshow(canny_image)
plt.show()