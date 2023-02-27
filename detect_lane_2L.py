import cv2
import numpy as np
from pythonDetect.Lane_detect import LaneDetector

# line detection
ld = LaneDetector()

# create mask polygons
def region_of_interest_left(image):
    height = image.shape[0]
    width = image.shape[1]

    arr = []
    arr.append([(0, height), (100, height), (600, 470), (350, 470), (0, height-100)])
    polygons = np.array(arr)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def average_slope_intercept_left ( image, lines):
    left_fit = []
    for line in lines:
        x1,y1,x2,y2 =line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
    
    left_fit_average = np.average(left_fit, axis=0)
    # left_fit_average = min(left_fit)
    left_line = abs(make_coordinates_left(image, left_fit_average))
        
    return left_line

def display_lines_left (image, lines):
    # display black
    line_image = np.zeros_like(image)
    temp = [0]
    arr = []
    if lines is not None:
        print("----------------------------")
        print(lines)
        for x1, y1, x2, y2 in lines: 
            if (700 > x1 > 0  and 700 > x2 > 0):
                arr.append([x1, y1])
                arr.append([x2, y2])

        if len(arr) == 4 and arr[0][0] < 300 and arr[1][0] < 600:
            temp = arr[2]; arr[2] = arr[3]; arr[3] = temp
            pts = np.array(arr, np.int32)
            cv2.fillPoly(line_image, [pts], (255,0,0))
    return line_image

def make_coordinates_left(image, line_parameters):
        # slope, intercept = line_parameters
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.001 ,0
        height = height = image.shape[0]
        y1 = height - 80
        y2 = int(y1*(7.3/10)) #y1*(2/4)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

# video = "video/road_car.mp4"
# video = "video/test2.mp4"
# video = "video/lane1.mp4"
video = "video/car_light6.mp4"

cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture("video/car_light6.mp4")
        continue

    frame = cv2.resize(frame, [1280, 720])
    canny_image = ld.canny(frame)
    cropped_image = ld.region_of_interest(canny_image.copy())
    cropped_image_left = region_of_interest_left(canny_image.copy())

    #detection
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=50)
    lines_left = cv2.HoughLinesP(cropped_image_left, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=50)

    if lines is not None:
        averaged_lines = ld.average_slope_intercept(frame, lines)
        averaged_lines_temp = averaged_lines[0]
        # threshold
        # line_image = display_lines(lane_image, lines)
        line_image = ld.display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    else:
        combo_image = frame.copy()

    # find left lane---------------------------------------------------------------------------------
    if (lines_left is not None) and lines is not None:
        averaged_lines_left = average_slope_intercept_left(frame, lines_left)
        combo_left = np.array([averaged_lines_left, averaged_lines_temp])
        # print(combo)
        line_image_left = display_lines_left(frame, combo_left)
        combo_image_left = cv2.addWeighted(line_image, 0.8, line_image_left, 1, 1)
        result = cv2.addWeighted(frame, 0.8, combo_image_left, 1, 1)
    else:
        result = frame.copy()


    cv2.imshow("cropped_image_left", cropped_image_left)
    cv2.imshow("cropped_image", cropped_image)
    cv2.imshow("result", result)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()