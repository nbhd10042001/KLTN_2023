import cv2 
import numpy as np
from pythonDetect.Lane_detect import LaneDetector
import os
import time

pathCurrent = os.path.dirname(__file__)
pathVideo = os.path.join(pathCurrent, "video")

# line detection
ld = LaneDetector()

video = pathVideo + "\lcl5.mp4"
vd = cv2.VideoCapture(video)
FPS = 60
start = time.time()
while(True):
    last = time.time()
    if last - start >  (1/FPS):
        start = last
        ret, frame = vd.read()
        if not ret:
            vd = cv2.VideoCapture(video)
            continue

        frame = cv2.resize(frame, [1280, 720])
        # frame = cv2.resize(frame, [640, 480])
        canny_image = ld.canny(frame)
        cropped_image = ld.region_of_interest(canny_image)

        #detection
        lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=10)
        
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
        vd.release()
        cv2.destroyAllWindows()
        break
