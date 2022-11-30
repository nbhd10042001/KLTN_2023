import cv2
import numpy as np
from Vehicle_detect import VehicleDetector
import matplotlib.pyplot as plt
import glob  

font = cv2.FONT_HERSHEY_COMPLEX

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


# Load vehicle detector
vd = VehicleDetector()

#load img
img = cv2.imread("img/car_lights/light2.jpg")
img = cv2.resize(img, [960, 640])
img_copy1 = img.copy()
img_copy2 = img.copy()

vehicle_boxes = vd.detect_vehicles(img_copy1)
# print (vehicle_boxes)
vehicle_count = len(vehicle_boxes)
crop_vehicle = region_of_interest(img_copy2, vehicle_boxes)

# detect vehicle
for box in vehicle_boxes:
    x, y, w, h = box
    cv2.rectangle(img_copy1, (x, y), (x + w, y + h), (255,0,0), 2)
    cv2.putText(img_copy1, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)

    h_i = img_copy1.shape[0]
    #calculate distance
    dis = ((h_i) / h)
    cv2.putText(img_copy1, "Dis: " + str(round(dis, 2)) + "m", (x, y), 0, 1, (255, 0, 0), 2)

# detect color lights
hsv = cv2.cvtColor(crop_vehicle, cv2.COLOR_BGR2HSV)
# find mask and threshold
lower = np.array([20, 111, 110])
upper = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower, upper)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# loop contuors
for cnt in contours:
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    # loc nhieu
    if area > 20:
        cv2.drawContours(img_copy1, [approx], 0, (0, 0, 0), 5)

cv2.imshow("img", img_copy1)
# cv2.imshow("crop", crop_vehicle)
# cv2.imshow("mask_crop", mask)
cv2.imshow("hsv", hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()