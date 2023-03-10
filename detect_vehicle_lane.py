import cv2
import numpy as np
from pythonDetect.Lane_detect import LaneDetector
from pythonDetect.Vehicle_detect import VehicleDetector
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
import time


# line detection
ld = LaneDetector()
# Load vehicle detector
vd = VehicleDetector_yolov5()

def crop_vehicle(image, boxs):
    arr = []
    for box in boxs:
        x, y, w, h, _ = box
        x = x - 20; y = y - 20
        w = w + 40; h = h + 40
        # add box to arr
        arr.append([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
    polygons = np.array(arr)
    # mask = np.zeros_like(image)
    mask = cv2.fillPoly(image, polygons, (0,0,0))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# video = "video/test2.mp4"
# video = "video/car1.mp4"
video = "video/car_light6.mp4"

cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(video)
        continue
    frame = cv2.resize(frame, [1280, 720])
    frame2 = frame.copy()

    # detect vehicle-----------------------------------------------------------------------------------------------
    vehicle_boxes, _ = vd.detect_vehicles(frame)
    # print (vehicle_boxes)
    vehicle_count = len(vehicle_boxes)

    # detect lines ------------------------------------------------------------------------------------------------
    frame2 = crop_vehicle(frame2, vehicle_boxes)
    canny_image = ld.canny(frame2)
    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, conf = box
            x_ca = x - 20; y_ca = y - 20
            w_ca = w + 40; h_ca = h + 40
            cv2.rectangle(canny_image, (x_ca, y_ca), (x_ca + w_ca, y_ca + h_ca), (0,0,0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(frame, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)
            cv2.putText(frame, "{:.2f}".format(conf), (x, y), 0, 0.5, (0, 255, 0), 2)

    cropped_image = ld.region_of_interest(canny_image)

    #detection
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=50)
    
    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame2, lines)
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame2, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        combo_image = frame.copy()

    end = time.time()
    print("frame/second", end - start)

    cv2.imshow("cropped_image", cropped_image)
    # cv2.imshow("canny_image", canny_image)
    cv2.imshow("result", combo_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
