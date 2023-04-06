import cv2
from PIL import ImageGrab
import time
import numpy as np
from pythonDetect.Lane_detect import LaneDetector

# line detection
ld = LaneDetector()
last_time = time.time()

while(True):
    frame = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[0], frame.shape[1]
    # frame = frame[0:int(h-(h/4)),0:w] # [y1:y2, x1:x2]
    frame = cv2.resize(frame, [1280, 720])

    canny_image = ld.canny(frame)
    cropped_image = ld.region_of_interest(canny_image)

    #detection
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=50)
    
    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        print(averaged_lines)
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 1, line_image, 0.5, 1)
    else:
        combo_image = frame.copy()


    cv2.imshow("cropped_image", cropped_image)
    # cv2.imshow("canny", canny_image)
    cv2.imshow("result", combo_image)
    print("Time: {}".format(time.time() - last_time))
    last_time = time.time()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break