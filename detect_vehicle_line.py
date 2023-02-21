import cv2
import numpy as np
from Line_detect import LineDetector
from Vehicle_detect import VehicleDetector

# line detection
ld = LineDetector()
# Load vehicle detector
vd = VehicleDetector()


# cap = cv2.VideoCapture("video/road_car.mp4")
# cap = cv2.VideoCapture("video/test2.mp4")
cap = cv2.VideoCapture("video/car_light3.mp4")



while(cap.isOpened()):
    _, frame = cap.read()
    frame = cv2.resize(frame, [1280, 720])

    # detect vehicle-----------------------------------------------------------------------------------------------
    vehicle_boxes = vd.detect_vehicles(frame)
    # print (vehicle_boxes)
    vehicle_count = len(vehicle_boxes)

    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(frame, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)


    # detect lines ------------------------------------------------------------------------------------------------
    canny_image = ld.canny(frame)
    cropped_image = ld.region_of_interest(canny_image)

    #detection
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        combo_image = frame.copy()


    # cv2.imshow("Car", frame)
    cv2.imshow("cropped_image", cropped_image)
    cv2.imshow("result", combo_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
