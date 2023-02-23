import cv2
import numpy as np
from Lane_detect import LaneDetector
from Vehicle_detect import VehicleDetector

# line detection
ld = LaneDetector()
# Load vehicle detector
vd = VehicleDetector()

def crop_vehicle(image, boxs):
    arr = []
    for i in range(len(boxs)):
        x, y, w, h = boxs[i]
        x = x - 20; y = y - 20
        w = w + 40; h = h + 40
        # add box to arr
        arr.append([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
    polygons = np.array(arr)
    # mask = np.zeros_like(image)
    mask = cv2.fillPoly(image, polygons, (0,0,0))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


cap = cv2.VideoCapture("video/road_car.mp4")
# cap = cv2.VideoCapture("video/test2.mp4")
# cap = cv2.VideoCapture("video/car1.mp4")
# cap = cv2.VideoCapture("video/car_light6.mp4")

while(cap.isOpened()):
    _, frame = cap.read()
    frame = cv2.resize(frame, [1280, 720])
    frame2 = frame.copy()

    # detect vehicle-----------------------------------------------------------------------------------------------
    vehicle_boxes = vd.detect_vehicles(frame)
    # print (vehicle_boxes)
    vehicle_count = len(vehicle_boxes)


    # detect lines ------------------------------------------------------------------------------------------------
    frame2 = crop_vehicle(frame2, vehicle_boxes)
    canny_image = ld.canny(frame2)
    for box in vehicle_boxes:
        x, y, w, h = box
        x_ca = x - 20; y_ca = y - 20
        w_ca = w + 40; h_ca = h + 40
        cv2.rectangle(canny_image, (x_ca, y_ca), (x_ca + w_ca, y_ca + h_ca), (0,0,0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(frame, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)

    cropped_image = ld.region_of_interest(canny_image)

    #detection
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame2, lines)
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame2, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        combo_image = frame.copy()


    cv2.imshow("cropped_image", cropped_image)
    # cv2.imshow("canny_image", canny_image)
    cv2.imshow("result", combo_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
