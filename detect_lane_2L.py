import cv2
import numpy as np
from pythonDetect.Lane_detect import LaneDetector
from pythonDetect.Lane_left_detect import LaneDetector_left
from pythonDetect.Lane_right_detect import LaneDetector_right
import time

# line detection
ld = LaneDetector()
lane_left = LaneDetector_left()
lane_right = LaneDetector_right()

# video = "video/road_car.mp4"
# video = "video/test2.mp4"
video = "video/lane1.mp4"
# video = "video/car_light6.mp4"

cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture("video/car_light6.mp4")
        continue
    
    start = time.time()
    frame = cv2.resize(frame, [1280, 720])
    canny_image = ld.canny(frame)
    cropped_image = ld.region_of_interest(canny_image.copy())
    cropped_image_left = lane_left.region_of_interest_left(canny_image.copy())
    cropped_image_right = lane_right.region_of_interest_right(canny_image.copy())

    #detection
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=50)
    lines_left = cv2.HoughLinesP(cropped_image_left, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=40)
    lines_right = cv2.HoughLinesP(cropped_image_right, 1, np.pi/180, 50, np.array([]), minLineLength=10, maxLineGap=40)

    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        averaged_lines_left_temp = averaged_lines[0]
        averaged_lines_right_temp = averaged_lines[1]
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame, averaged_lines)

        # find left lane---------------------------------------------------------------------------------
        if lines_left is not None:
            averaged_lines_left = lane_left.average_slope_intercept_left(frame, lines_left)
            combo_left = np.array([averaged_lines_left, averaged_lines_left_temp])
            # print(combo)
            line_image_left = lane_left.display_lines_left(frame, combo_left) # img lane da xu li
            combo_image_left = cv2.addWeighted(line_image, 0.8, line_image_left, 1, 1) # ket hop

            # find right lane---------------------------------------------------------------------------------
            if (lines_right is not None):
                averaged_lines_right = lane_right.average_slope_intercept_right(frame, lines_right)
                combo_right = np.array([averaged_lines_right_temp, averaged_lines_right])
                # print(combo_right)
                line_image_right = lane_right.display_lines_right(frame, combo_right)
                combo_image_right = cv2.addWeighted(combo_image_left, 0.8, line_image_right, 1, 1)
                result = cv2.addWeighted(frame, 0.8, combo_image_right, 1, 1)
            else:
                result = cv2.addWeighted(frame, 0.8, combo_image_left, 1, 1)
        else:
            result = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        result = frame.copy()

    end = time.time()
    print(end - start)

    # cv2.imshow("cropped_image_left", cropped_image_left)
    # cv2.imshow("cropped_image_right", cropped_image_right)
    # cv2.imshow("cropped_image", cropped_image)
    cv2.imshow("result", result)
    cv2.imshow("canny_image", canny_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()