import cv2
import time
import numpy as np
import os
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
from pythonDetect.Light_detect_and_Warning import LightSignal_and_Warnings
from pythonDetect.Lane_detect import LaneDetector

# Load path
pathFile = os.path.dirname(__file__)
pathVideo = os.path.join(pathFile, 'video')
# mp4 = pathVideo + "/car/car3_Trim.mp4"
mp4 = pathVideo + "/slow_traffic_small.mp4"

# Load file import
vd = VehicleDetector_yolov5()
ld = LaneDetector()
detect_light_warning = LightSignal_and_Warnings()

# Load video
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

# Loop through the images
while True:
    start = time.time()
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture(0)
        continue

    if(type(frame) == type(None)):
        pass
    else:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    frame1 = frame.copy()
    vehicle_boxes, _, light_boxs = vd.detect_vehicles(frame)
    vehicle_count = len(vehicle_boxes) # find number car
    vbox_lags = []
    classCar = []
    lightBoxs = []
    cv2.putText(frame, "Number of Vehicles: " + str(vehicle_count), (20, 50), 0, 1, (0, 255, 0), 2)
    height, width = frame.shape[0], frame.shape[1]
    mask = np.zeros_like(frame)
    masked_image = cv2.bitwise_and(frame, mask)

# - detect lane
    canny_image = ld.canny(frame)
    region_image = ld.region_of_interest(canny_image)
    # perspective_trans, Minv = ld.perspective_transform(frame)
    centCamera = [[int(width/2), height],[int(width/2), int(height*0.95)]]
    cv2.line(frame, centCamera[0], centCamera[1], (255, 0, 0), 2)

    lines = cv2.HoughLinesP(region_image, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=10)
    if lines is not None and len(lines) < 70:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        line_image, arrayLines, centerPoint = ld.display_lines(frame, averaged_lines)
        # print(arrayLines) # 4 diem lien tiep tao thanh da giac
        if len(arrayLines) == 4 and centerPoint == True:
            # draw center point
            centerPoint_x = int((arrayLines[2][0] - arrayLines[1][0])/2) + arrayLines[1][0]
            point1 = [centerPoint_x, height]
            point2 = [centerPoint_x, int(height*0.95)]
            cv2.line(line_image, point1, point2, (0, 255, 255), 2)

            # draw center line 1 and 2
            centerLine1 = int((arrayLines[1][0] - arrayLines[0][0])/2) + arrayLines[0][0]
            centerLine2 = int((arrayLines[3][0] - arrayLines[2][0])/2) + arrayLines[2][0]
            p_ctLine1_x = [centerLine1, height]
            p_ctLine1_y = [centerLine1, int(height*0.95)]
            p_ctLine2_x = [centerLine2, height]
            p_ctLine2_y = [centerLine2, int(height*0.95)]
            cv2.line(line_image, p_ctLine1_x, p_ctLine1_y, (0, 255, 255), 2)
            cv2.line(line_image, p_ctLine2_x, p_ctLine2_y, (0, 255, 255), 2)

            # kiem tai xe co vuot lan
            ctCamera_x = centCamera[0][0]
            if centerLine1 < ctCamera_x < centerLine2:
                cv2.line(masked_image, arrayLines[0], arrayLines[1], (0, 0, 255), 5) # draw line on masked_image
                cv2.line(masked_image, arrayLines[2], arrayLines[3], (0, 0, 255), 5)
                if ctCamera_x <= centerLine1:
                    cv2.putText(line_image, "Xe re trai", (20, 300), 0, 2, (0, 255, 0), 2)
                if ctCamera_x >= centerLine2:
                    cv2.putText(line_image, "Xe re phai", (20, 300), 0, 2, (0, 255, 0), 2)

# - Detect car and lane crossing warning
    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, conf = box
            frame, masked_image = detect_light_warning.lane_crossing_warning(frame, box, masked_image)
            # find lag box 
            if w > int(width*0.05) and (w < int(h*2)):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                vbox_lags.append(box)
                car = Car(x, y, w, h)
                classCar.append(car)
            else: cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2) # small box car

# - Detect light car
    if light_boxs:
        for light_box in light_boxs:
            x, y, w, h, conf = light_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.putText(frame,"{:.2f}".format(conf), (x, y), 0, 0.5, (0, 0, 255), 1)


# - detect color lights
    crop_lights = detect_light_warning.crop_lights_vehicle(frame1, vbox_lags)
    mask = detect_light_warning.create_mask_hsv(crop_lights)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for cnt in contours:
            xl, yl, wl, hl = cv2.boundingRect(cnt)
            lightBoxs.append([xl, yl, wl, hl])
        frame = detect_light_warning.handle_lightSignal(frame, classCar, lightBoxs)

    classCar = []
    result = cv2.addWeighted(frame, 1, line_image, 0.5, 1)
    end = time.time()
    cv2.putText(result,"fps: {:.3f}s".format(end - start), (int(width - width/4), 50), 0, 1, (255, 0, 0), 2)
    
    cv2.imshow("mask_image", masked_image)
    cv2.imshow("result", result)
    # cv2.imshow("mask_crop_light", crop_lights)
    # cv2.imshow("frame", frame)

# - press key Q to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
