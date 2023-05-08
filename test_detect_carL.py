import cv2
import time
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
import numpy as np
import os

pathFile = os.path.dirname(__file__)
pathVideo = os.path.join(pathFile, 'video')
mp4 = pathVideo + "/car/car3_Trim.mp4"

# Load vehicle detector
vd = VehicleDetector_yolov5()
video = cv2.VideoCapture(mp4)
# video = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

class Car():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.turnRight = False
        self.turnLeft = False

def crop_lights_vehicle(image, boxs):
    arr = []
    for i in range(len(boxs)):
        x, y, w, h, cf = boxs[i]
        # add box to arr
        h14 = int(h/4)
        w13 = int(w/3)
        # arr.append([(x, y + h2), (x + w13, y + h2), (x + w13, y+h), (x, y+h)])
        # arr.append([(x + w33, y + h2), (x + w, y + h2), (x + w, y+h), (x + w33, y+h)])
        arr.append([(x, y+h14), (x + w13, y+h14), (x + w13, y+h), (x, y+h)])
        arr.append([((x+w)-w13, y+h14), (x + w, y+h14), (x + w, y+h), ((x+w)-w13, y+h)])
    polygons = np.array(arr)
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, polygons, (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# Loop through the images
while True:
    start = time.time()
    ret, frame = video.read()
    frame = cv2.resize(frame, [640 ,480])
    if not ret:
        cap = cv2.VideoCapture(0)
        continue

    frame1 = frame.copy()
    vehicle_boxes, _, _ = vd.detect_vehicles(frame)
    vehicle_count = len(vehicle_boxes) # find number car
    vbox_lags = []
    classCar = []
    lightBoxs = []
    cv2.putText(frame, "Number of Vehicles: " + str(vehicle_count), (20, 50), 0, 1, (0, 255, 0), 2)
    height, width = frame.shape[0], frame.shape[1]

    arr = []
    p1 = (int(width/8), height)
    p2 = (int(width/2 - width/35 + 10), int(height/2 - 50))
    p3 = (int(width/2 + width/35 - 40), int(height/2 - 50))
    p4 = (int(width - width/8), height)

    arr.append([p1, p2, p3, p4])
    pts = np.array(arr, np.int32)
    mask = np.zeros_like(frame)
    masked_image = cv2.bitwise_and(frame, mask)
    cv2.line(masked_image, p1, p2, (0, 0, 255), 15)
    cv2.line(masked_image, p3, p4, (0, 0, 255), 15)


    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, conf = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(frame,"{:.2f}".format(conf), (x, y), 0, 0.5, (255, 255, 0), 1)

            # Warning vehicle crossing the lane ===============================================================
            center = [int(x + w/2), int(y + h)] #center box of vehicle
            cv2.circle(masked_image, center, 5, 255, 2)
            b,g,r = masked_image[center[1], center[0]]

            # find lag box ----------------------------------------------------------------
            if w > int(width/25) and (w < int(h*2)):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                vbox_lags.append(box)
                car = Car(x, y, w, h)
                classCar.append(car)

            if (b == 0 and g == 0 and r == 255):
                cv2.putText(frame,"Warning! Co xe vuot lan", (20, 110), 0, 1, (0, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    crop_lights = crop_lights_vehicle(frame1, vbox_lags)
    # detect color lights------------------------------------------------------------------------------------------------
    hsv = cv2.cvtColor(crop_lights, cv2.COLOR_BGR2HSV)
    # find mask and threshold
    lower = np.array([14, 140, 140])
    upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = cv2.dilate(mask, kernel, iterations=8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if find contours (True)
    if contours:
        # loop contuors-------------------------------------------------------------------
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            # loc nhieu
            if area > 50:
                xl, yl, wl, hl = cv2.boundingRect(cnt)
                lightBoxs.append([xl, yl, wl, hl])
        
        for car in classCar:
            for lightBox in lightBoxs:
                xl, yl, wl, hl = lightBox
                x_cent = xl + int(wl/2)
                y_cent = yl + int(hl/2)

                # detect turn signal lights 
                if (car.x <= x_cent <= (car.x + int((car.w)/3))) and ((car.y) <= y_cent <= (car.y + car.h)):
                    car.turnLeft = True
                    cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                    cv2.putText(frame, "Left", (xl, yl), 0, 0.5, (0, 255, 255), 2)

                if ((car.x + car.w)- int((car.w)/3)) <= x_cent <= (car.x + car.w) and (car.y) <= y_cent <= (car.y + car.h):
                    car.turnRight = True
                    cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                    cv2.putText(frame, "Right", (xl, yl), 0, 0.5, (0, 255, 255), 2)
    
            if car.turnRight == True and car.turnLeft == True:
                cv2.putText(frame,"Warning! Xe dung!", (20, 80), 0, 1, (0, 255, 255), 2)
            if car.turnRight == True and car.turnLeft == False:
                cv2.putText(frame,"Warning! Co den xi nhan Phai", (20, 80), 0, 1, (0, 255, 255), 2)
            if car.turnRight == False and car.turnLeft == True:
                cv2.putText(frame,"Warning! Co den xi nhan Trai", (20, 80), 0, 1, (0, 255, 255), 2)

    classCar = []
    result = cv2.addWeighted(frame, 1, masked_image, 0.3, 1)

    end = time.time()
    cv2.putText(result,"fps: {:.3f}s".format(end - start), (int(width - width/4), 50), 0, 1, (255, 0, 0), 2)
    
    cv2.imshow("mask", masked_image)
    cv2.imshow("result", result)
    cv2.imshow("mask_crop_light", crop_lights)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
