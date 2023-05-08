import cv2 
import numpy as np
from pythonDetect.Lane_detect import LaneDetector
import os
import time

pathCurrent = os.path.dirname(__file__)
pathVideo = os.path.join(pathCurrent, "video")

# line detection
ld = LaneDetector()

video = pathVideo + "\lane2.mp4"
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
        h, w = frame.shape[0], frame.shape[1]

        canny_image = ld.canny(frame)
        cropped_image = ld.region_of_interest(canny_image)

        # perspective_trans, Minv = ld.perspective_transform(frame)
        centCamera = [[int(w/2), h],[int(w/2), int(h*0.95)]]
        cv2.line(frame, centCamera[0], centCamera[1], (255, 0, 0), 5)

        #detection
        lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=10)
        try:
            print(len(lines))
        except:
            print("Error")
        if lines is not None and len(lines) < 70:
            averaged_lines = ld.average_slope_intercept(frame, lines)
            # threshold
            # line_image = display_lines(lane_image, lines)
            line_image, arrayLines, centerLine = ld.display_lines(frame, averaged_lines)

            if len(arrayLines) == 4 and centerLine == True:
                centerLine = int((arrayLines[2][0] - arrayLines[1][0])/2) + arrayLines[1][0]
                point1 = [centerLine, h]
                point2 = [centerLine, int(h*0.95)]
                cv2.line(line_image, point1, point2, (0, 0, 255), 5)
                if centerLine < w/2:
                    cv2.putText(line_image, "Xe re trai", (20, 50), 0, 2, (0, 255, 0), 2)
                else:
                    cv2.putText(line_image, "Xe re phai", (20, 50), 0, 2, (0, 255, 0), 2)

            combo_image = cv2.addWeighted(frame, 1, line_image, 0.5, 1)
        else:
            combo_image = frame.copy()


        cv2.imshow("cropped_image", cropped_image)
        cv2.imshow("result", combo_image)
        # cv2.imshow("perspective_trans", perspective_trans)

    key = cv2.waitKey(1)
    if key == ord('q'):
        vd.release()
        cv2.destroyAllWindows()
        break
