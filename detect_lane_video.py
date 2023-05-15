import cv2 
import numpy as np
from pythonDetect.Lane_detect import LaneDetector
import os
import time

pathCurrent = os.path.dirname(__file__)
pathVideo = os.path.join(pathCurrent, "video/")
video = pathVideo + "lane/ok4.mp4"
video = pathVideo + "lane4.mp4"
vd = cv2.VideoCapture(video)

# line detection
ld = LaneDetector()
FPS = 60
start = time.time()
ScaleAbs_high = False
ScaleAbs_low = False
text_ScaleAbs = "None"
color_selection = "HLS" 

while True:
    last = time.time()
    if last - start >  (1/FPS):
        start = last
        ret, frame = vd.read()
        if not ret:
            vd = cv2.VideoCapture(video)
            continue
        if ScaleAbs_high == True:
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=5)
        if ScaleAbs_low == True:
            frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=5)

        frame = cv2.resize(frame, [640, 480])
        height, width = frame.shape[:2]
        if color_selection == "RGB":
            frame_color = ld.RGB_color_selection(frame)
        if color_selection == "HSV":
            frame_color = ld.HSV_color_selection(frame)
        if color_selection == "HLS":
            frame_color = ld.HLS_color_selection(frame)

        cv2.putText(frame,"color selection: "+ color_selection, (int(width*0.1), 50),
                         0, 0.5, (255, 0, 0), 2)
        cv2.putText(frame,"ScaleAbs: "+ text_ScaleAbs, (int(width*0.1), 70),
                         0, 0.5, (255, 0, 0), 2)
        
        canny_image = ld.canny(frame_color)
        region_image = ld.region_of_interest(canny_image)

        #detection
        lines = cv2.HoughLinesP(region_image, rho = 1, theta = (np.pi/180), threshold = 20,
                           minLineLength = 20, maxLineGap = 300)

        if lines is not None:
            averaged_lines = ld.average_slope_intercept(frame, lines)
            print(averaged_lines)
            line_image, _, _ = ld.display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 1, line_image, 0.5, 1)
        else:
            combo_image = frame.copy()

        cv2.imshow("region_image", region_image)
        cv2.imshow("canny_image", canny_image)
        cv2.imshow("result", combo_image)

    key = cv2.waitKey(1)
    if key == ord('1'):
        color_selection = "RGB"
    if key == ord('2'):
        color_selection = "HSV"
    if key == ord('3'):
        color_selection = "HLS"

    if key == ord('h'):
        ScaleAbs_high = True
        ScaleAbs_low = False
        text_ScaleAbs = "High"
    if key == ord('l'):
        ScaleAbs_low = True
        ScaleAbs_high = False
        text_ScaleAbs = "Low"
    if key == ord('n'):
        ScaleAbs_high = False
        ScaleAbs_low = False
        text_ScaleAbs = "None"
    if key == ord('q'):
        vd.release()
        cv2.destroyAllWindows()
        break

