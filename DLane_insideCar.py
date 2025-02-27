import cv2 
import numpy as np
from pythonDetect.Lane_detectA import LaneDetector

# line detection
ld = LaneDetector()

# video = "video/lane3.mp4"
video = "video/lcl7.mp4"

cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(video)
        continue

    frame = cv2.resize(frame, [1280, 720])
    # frame = cv2.resize(frame, [640, 480])
    canny_image = ld.canny(frame)
    cropped_image = ld.region_of_interest(canny_image)

    #detection
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=100)
    
    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 1, line_image, 0.5, 1)
    else:
        combo_image = frame.copy()


    cv2.imshow("cropped_image", cropped_image)
    cv2.imshow("result", combo_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()