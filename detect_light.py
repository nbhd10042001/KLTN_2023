import cv2
import numpy as np
from Vehicle_detect import VehicleDetector
import matplotlib.pyplot as plt
import glob  

font = cv2.FONT_HERSHEY_COMPLEX

l_s = 0
l_v = 0

def region_of_interest(image, boxs):
    arr = []
    for i in range(len(boxs)):
        x, y, w, h = boxs[i]
        arr.append([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
    polygons = np.array(arr)
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, polygons, (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

img = cv2.imread("img/car/first_person_car1.jpg")

# Load vehicle detector
vd = VehicleDetector()
img = cv2.resize(img, [960, 640])

vehicle_boxes = vd.detect_vehicles(img)
# print (vehicle_boxes)
vehicle_count = len(vehicle_boxes)
cut_car = region_of_interest(img, vehicle_boxes)

for box in vehicle_boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
    cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)

    h_i = img.shape[0]
    #calculate distance
    dis = ((h_i) / h)
    cv2.putText(img, "Dis: " + str(round(dis, 2)) + "m", (x, y), 0, 1, (255, 0, 0), 2)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

l_s, l_v = 140, 140
hue_value = 20

# find mask and threshold
lower = np.array([hue_value, l_s, l_v])
upper = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower, upper)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.drawContours(img, (approx), 0, (0,0,0), 1)
    # print(x, y, w, h)
    # approximate the contour

cv2.imshow("img", img)
cv2.imshow("mask", mask)
cv2.imshow("cut", cut_car)
cv2.waitKey(0)

cv2.destroyAllWindows()