import cv2 
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
import time
# import matplotlib.pyplot as plt
import glob  
import numpy as np

# video = cv2.VideoCapture("video\car\car2.mp4")
# video = cv2.VideoCapture("video\light_blink\light_blink6.mp4")
video = cv2.VideoCapture("video\lcl.mp4")
# video = cv2.VideoCapture(0)

# video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# video.set(10, 0)
# video.set(11, 0)
font = cv2.FONT_HERSHEY_COMPLEX

def crop_lights_vehicle(image, boxs):
    arr = []
    for i in range(len(boxs)):
        x, y, w, h, cf = boxs[i]
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
vd = VehicleDetector_yolov5()
last_time = time.time()
while True:
    #load video
    _, frame = video.read(0)
    # frame = cv2.resize(frame, [640, 480])
    height, width = frame.shape[0], frame.shape[1]

    frame1 = frame.copy()

    vehicle_boxes, _ = vd.detect_vehicles(frame)
    vehicle_count = len(vehicle_boxes)
    vbox_lag = []

    # crop_v_gray = cv2.cvtColor(crop_vehicle, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(crop_v_gray, 200, 255, cv2.THRESH_BINARY)

    # detect vehicle ------------------------------------------------------------------------------------------------
    for box in vehicle_boxes:
        x, y, w, h, cf = box
        # print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(frame, "Vehicles: " + str(vehicle_count), (20, 50), 0, 1, (0, 255, 0), 2)

        # find lag box ----------------------------------------------------------------
        if w > int(width/25) and (w < int(h*2)):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            vbox_lag.append(box)


    crop_lights = crop_lights_vehicle(frame1, vbox_lag)
    # detect color lights------------------------------------------------------------------------------------------------
    hsv = cv2.cvtColor(crop_lights, cv2.COLOR_BGR2HSV)
    # find mask and threshold
    lower = np.array([14, 140, 140])
    upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    bitw = cv2.bitwise_and(frame, frame, mask=mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = cv2.dilate(mask, kernel, iterations=8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if find contours (True)
    if contours:
        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        # cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        x1_center = x1 + int(w1/2)
        y1_center = y1 + int(h1/2)

        # detect turn signal lights 
        for box in vbox_lag:
            x, y, w, h, cf = box
            if (x <= x1_center <= (x + int(w/3))) and ((y) <= y1_center <= (y + h)):
                cv2.rectangle(frame, (x1 ,y1), (x1 + w1, y1 + h1), (0,255,255), 1)
                cv2.putText(frame, "Left", (x1, y1), 0, 0.5, (0, 255, 255), 2)
                cv2.putText(frame,"Warning!", (20, 80), 0, 1, (0, 255, 255), 2)

            elif ((x + w)- int(w/3)) <= x1_center <= (x + w) and (y) <= y1_center <= (y + h):
                cv2.rectangle(frame, (x1 ,y1), (x1 + w1, y1 + h1), (0,255,255), 1)
                cv2.putText(frame, "Right", (x1, y1), 0, 0.5, (0, 255, 255), 2)
                cv2.putText(frame,"Warning!", (20, 80), 0, 1, (0, 255, 255), 2)


    cv2.imshow("img", frame)
    # cv2.imshow("crop", crop_lights)
    cv2.imshow("mask_crop", mask)

    print("Time: {}".format(time.time() - last_time))
    last_time = time.time()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
