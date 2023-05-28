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
# mp4 = pathVideo + "/lane/2light.mp4"
mp4 = pathVideo + "/lane/ok6.mp4"
# mp4 = pathVideo + "/lane4.mp4"

# Load file import
vd = VehicleDetector_yolov5()
ld = LaneDetector()
detect_light_warning = LightSignal_and_Warnings()

# Load video
video = cv2.VideoCapture(mp4)
# video = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
text_ScaleAbs = "None"
color_selection = 3
color_selec_text = ""
fps_count = 0
fps = 0

class Car():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.turnRight = False
        self.turnLeft = False
        self.numberLight = 0

def crop_vehicle(image, boxs):
    arr = []
    for box in boxs:
        x, y, w, h, _ = box
        # add box to arr
        arr.append([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
    polygons = np.array(arr)
    # mask = np.zeros_like(image)
    mask = cv2.fillPoly(image, polygons, (0,0,0))
    masked_image_crop = cv2.bitwise_and(image, mask)
    return masked_image_crop

def changeScaleAbs(frame, text_ScaleAbs):
    text_ScaleAbs = "None"
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 100], dtype="uint8")
    upper = np.array([80, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    brightness = np.average(mask)
    # print(brightness)
    if brightness > 140:
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=5)
        text_ScaleAbs = "Low"
    if brightness < 2:
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=5)
        text_ScaleAbs = "High"
    return frame, text_ScaleAbs

start = time.time()
while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture(0)
        continue
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[0], frame.shape[1]
    frame, text_ScaleAbs = changeScaleAbs(frame.copy(), text_ScaleAbs)
    
    if color_selection == 1:
        frame_color = ld.RGB_color_selection(frame.copy())
        color_selec_text = "RGB"
    if color_selection == 2:
        frame_color = ld.HSV_color_selection(frame.copy())
        color_selec_text = "HSV"
    if color_selection == 3:
        frame_color = ld.HLS_color_selection(frame.copy())
        color_selec_text = "HLS"

    cv2.putText(frame,"ScaleAbs: "+ text_ScaleAbs, (int(width - width/4), 40),
                        0, 0.5, (255, 0, 0), 2)
    cv2.putText(frame,"Color Select: "+ color_selec_text, (int(width - width/4), 60),
                         0, 0.5, (255, 0, 0), 2)

    frame1 = frame.copy()
    vehicle_boxes, bike_boxes = vd.detect_vehicles(frame)
    vboxes_near = []
    classCar = []
    lightBoxes = []
    mask = np.zeros_like(frame)
    masked_image = cv2.bitwise_and(frame, mask)

# - detect lane
    canny_image = ld.canny(frame_color)
    # crop box car to detect lane
    canny_image = crop_vehicle(canny_image, vehicle_boxes)
    canny_image = crop_vehicle(canny_image, bike_boxes)

    region_image = ld.region_of_interest(canny_image)
    lines = cv2.HoughLinesP(region_image, rho = 1, theta = (np.pi/180), threshold = 20,
                           minLineLength = 20, maxLineGap = 300)
    # print(len(lines))
    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        line_image, arrayLines, two_lines, one_line = ld.display_lines(frame, averaged_lines)
        
        if len(arrayLines) == 4 and two_lines == True:
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

            #draw mask img
            cv2.line(masked_image, arrayLines[0], arrayLines[1], (0, 0, 255), 10)
            cv2.line(masked_image, arrayLines[2], arrayLines[3], (0, 0, 255), 10)

        if len(arrayLines) == 2 and one_line == True:
            cv2.line(masked_image, arrayLines[0], arrayLines[1], (0, 0, 255), 10) # draw line on masked_image

            
# - Detect car and lane crossing warning
    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, conf = box
            # find lag box 
            if w > int(width*0.05) and (w < int(h*2)) and (h < int(w*2)):
                vboxes_near.append(box)
                car = Car(x, y, w, h)
                classCar.append(car)
            else: cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2) # small box car   
    if vboxes_near:
        for box in vboxes_near:   
            x, y, w, h, conf = box     
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            frame, masked_image = detect_light_warning.lane_crossing_warning(frame, box, masked_image)

# - detect color lights
    crop_lights = detect_light_warning.crop_lights_vehicle(frame1, vboxes_near)
    mask_light = detect_light_warning.create_mask_hsv(crop_lights)
    contours, _ = cv2.findContours(mask_light, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for cnt in contours:
            xl, yl, wl, hl = cv2.boundingRect(cnt)
            lightBoxes.append([xl, yl, wl, hl])
        frame = detect_light_warning.handle_lightSignal(frame, classCar, lightBoxes)

    classCar = []
    if lines is not None:
        result = cv2.addWeighted(frame, 1, line_image, 0.3, 1)
    else: result = frame
        
# - caculate FPS 
    end = time.time()
    fps_count += 1
    if (end - start) >= 1:
        start = end
        fps = fps_count
        cv2.putText(result,"FPS: {}".format(fps_count), (int(width - width/4), 20), 0, 0.5, (255, 0, 0), 2)
        fps_count = 0
    else:
        cv2.putText(result,"FPS: {}".format(fps), (int(width - width/4), 20), 0, 0.5, (255, 0, 0), 2)

# - Show results
    # cv2.imshow("mask_image", masked_image)
    # cv2.imshow("region_image", region_image)
    # cv2.imshow("canny_image", canny_image)
    cv2.imshow("mask_crop_light", crop_lights)
    cv2.imshow("mask_light", mask_light)
    # cv2.imshow("frame", frame)
    cv2.imshow("result", result)

# - press key to select options
    key = cv2.waitKey(1)
    if key == ord('c'):
        color_selection += 1
        if color_selection > 3:
            color_selection = 1

    if key == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break
