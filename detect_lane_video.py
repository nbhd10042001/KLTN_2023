import cv2 
import numpy as np
from Lane_detect import LaneDetector

# line detection
ld = LaneDetector()

cap = cv2.VideoCapture("video/road_car.mp4")
# cap = cv2.VideoCapture("video/test2.mp4")
# cap = cv2.VideoCapture("video/car_light6.mp4")

while(cap.isOpened()):
    _, frame = cap.read()
    frame = cv2.resize(frame, [1280, 720])
    canny_image = ld.canny(frame)
    cropped_image = ld.region_of_interest(canny_image)

    #detection
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=50)
    
    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        combo_image = frame.copy()


    cv2.imshow("cropped_image", cropped_image)
    cv2.imshow("result", combo_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()