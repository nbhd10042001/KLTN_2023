import numpy as np
import cv2
import os
import glob

pathCurrent = os.path.dirname(__file__)
pathVideo = os.path.join(pathCurrent, "video/")

video = pathVideo + "lane/ok2.mp4"
video = pathVideo + "lane3.mp4"
# video = pathVideo + "lcl7.mp4"
vd = cv2.VideoCapture(video)

def RGB_color_selection(image):
    """
    Apply color selection to RGB images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    #White color mask
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    #Yellow color mask
    lower_threshold = np.uint8([175, 175, 0])
    upper_threshold = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    #Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image

def HSV_color_selection(image):
    #Convert the input image to HSV
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #White color mask
    lower_threshold = np.uint8([0, 0, 150])
    upper_threshold = np.uint8([255, 20, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Yellow color mask
    lower_threshold = np.uint8([11, 80, 80])
    upper_threshold = np.uint8([30, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    return masked_image

def HLS_color_selection(image):
    #Convert the input image to HSL
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    #White color mask
    lower_threshold = np.uint8([0, 160, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Yellow color mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    return masked_image

def region_selection(image):
    mask = np.zeros_like(image)   
    height, width = image.shape[:2]
    point_1 = [width * 0.1, height * 0.9]
    point_2 = [width * 0.4, height * 0.7]
    point_3 = [width * 1, height * 0.9]
    point_4 = [width * 0.7, height * 0.7]
    polygon = np.array([[point_1, point_2, point_4, point_3]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines, color = [255, 0, 0], thickness = 2):
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def average_slope_intercept(lines):
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.7
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness. 
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

ScaleAbs_high = False
ScaleAbs_low = False

while(True):
    ret, frame = vd.read()
    if not ret:
        vd = cv2.VideoCapture(video)
        continue
    if ScaleAbs_high == True:
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=5)
    if ScaleAbs_low == True:
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=5)

    frame = cv2.resize(frame, [640,480])
    frame_RGB = RGB_color_selection(frame)
    frame_HSV = HSV_color_selection(frame)
    frame_HLS = HLS_color_selection(frame)
    
    gray_RGB = cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2GRAY)
    gray_HSV = cv2.cvtColor(frame_HSV, cv2.COLOR_RGB2GRAY)
    gray_HLS = cv2.cvtColor(frame_HLS, cv2.COLOR_RGB2GRAY)

    blur_RGB = cv2.GaussianBlur(gray_RGB, (13, 13), 0)
    blur_HSV = cv2.GaussianBlur(gray_HSV, (13, 13), 0)
    blur_HLS = cv2.GaussianBlur(gray_HLS, (13, 13), 0)

    canny_RGB = cv2.Canny(blur_RGB, 50, 150)
    canny_HSV = cv2.Canny(blur_HSV, 50, 150)
    canny_HLS = cv2.Canny(blur_HLS, 50, 150)

    region_RGB = region_selection(canny_RGB)
    region_HSV = region_selection(canny_HSV)
    region_HLS = region_selection(canny_HLS)

    hough_lines_RGB = cv2.HoughLinesP(region_RGB, rho = 1, theta = (np.pi/180), threshold = 20,
                           minLineLength = 20, maxLineGap = 300)
    hough_lines_HSV = cv2.HoughLinesP(region_HSV, rho = 1, theta = (np.pi/180), threshold = 20,
                            minLineLength = 20, maxLineGap = 300)
    hough_lines_HLS = cv2.HoughLinesP(region_HLS, rho = 1, theta = (np.pi/180), threshold = 20,
                            minLineLength = 20, maxLineGap = 300)
    
    # line_image = draw_lines(frame, hough_lines_HLS)

    lane_image = draw_lane_lines(frame, lane_lines(frame, hough_lines_HLS))
    # cv2.imshow('frame_RGB',frame_RGB)
    # cv2.imshow('frame_HSV',frame_HSV)n
    # cv2.imshow('frame_HSL',frame_HSL)
    cv2.imshow('lane_image',lane_image)

    cv2.imshow('frame',frame)

    key = cv2.waitKey(1)
    if key == ord('h'):
        ScaleAbs_high = True
        ScaleAbs_low = False
    if key == ord('l'):
        ScaleAbs_low = True
        ScaleAbs_high = False
    if key == ord('n'):
        ScaleAbs_high = False
        ScaleAbs_low = False
    if key == ord('q'):
        vd.release()
        cv2.destroyAllWindows()
        break
