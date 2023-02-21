import cv2 
from Vehicle_detect import VehicleDetector
# import matplotlib.pyplot as plt
import glob  
import numpy as np


video = cv2.VideoCapture("video\car_light3_Trim.mp4")
video = cv2.VideoCapture("video\car_light1.mp4")
# video = cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(10, 0)
video.set(11, 0)
font = cv2.FONT_HERSHEY_COMPLEX

def crop_lights_vehicle(image, boxs):
    arr = []
    for i in range(len(boxs)):
        x, y, w, h = boxs[i]
        # add box to arr
        h2 = int(h/2)
        w13 = int(w/3)
        w33 = int((2*w)/3)
        # arr.append([(x, y + h2), (x + w13, y + h2), (x + w13, y+h), (x, y+h)])
        # arr.append([(x + w33, y + h2), (x + w, y + h2), (x + w, y+h), (x + w33, y+h)])
        arr.append([(x, y), (x + w13, y), (x + w13, y+h), (x, y+h)])
        arr.append([(x + w33, y), (x + w, y), (x + w, y+h), (x + w33, y+h)])
    polygons = np.array(arr)
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, polygons, (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# Load vehicle detector
vd = VehicleDetector()

while True:
    #load video
    _, frame = video.read(0)
    frame = cv2.resize(frame, [1280, 720])

    img_copy1 = frame.copy()
    img_copy2 = frame.copy()

    vehicle_boxes = vd.detect_vehicles(img_copy1)
    # print (vehicle_boxes)
    vehicle_count = len(vehicle_boxes)
    crop_lights = crop_lights_vehicle(img_copy2, vehicle_boxes)

    # crop_v_gray = cv2.cvtColor(crop_vehicle, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(crop_v_gray, 200, 255, cv2.THRESH_BINARY)

    # detect vehicle ------------------------------------------------------------------------------------------------
    for box in vehicle_boxes:
        x, y, w, h = box
        # print(x, y, w, h)
        cv2.rectangle(img_copy1, (x, y), (x + w, y + h), (255,0,0), 1)
        cv2.putText(img_copy1, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 1)

        h_i = img_copy1.shape[0]
        #calculate distance
        dis = ((h_i) / h)
        cv2.putText(img_copy1, "Dis: " + str(round(dis, 2)) + "m", (x, y), 0, 1, (255, 0, 0), 1)

    # detect color lights------------------------------------------------------------------------------------------------
    hsv = cv2.cvtColor(crop_lights, cv2.COLOR_BGR2HSV)

    
    # find mask and threshold
    lower = np.array([20, 140, 140])
    upper = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if find contours (True)
    if contours:
        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        # cv2.drawContours(img_copy1, [approx], 0, (0, 0, 0), 5)
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        # print(x1, y1, w1, h1)
        cv2.rectangle(img_copy1, (x1 ,y1), (x1 + w1, y1 + h1), (0,255,0), 1)

        # detect turn signal lights 
        for box in vehicle_boxes:
            x, y, w, h = box
            if x <= x1 <= (x + w/3) and (y) <= y1 <= (y + h):
                cv2.putText(img_copy1, "Left", (x1, y1), 0, 1, (0, 255, 0), 1)
            elif (x + (2*w)/3) <= x1 <= (x + w) and (y) <= y1 <= (y + h):
                cv2.putText(img_copy1, "Right", (x1, y1), 0, 1, (0, 255, 0), 1)


    cv2.imshow("img", img_copy1)
    # cv2.imshow("crop", crop_lights)
    cv2.imshow("mask_crop", mask)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
