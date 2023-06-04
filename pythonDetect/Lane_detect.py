import cv2
import numpy as np
import warnings    
warnings.simplefilter('ignore', np.RankWarning) 

class LaneDetector:
    def RGB_color_selection(self, image):
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

    def HSV_color_selection(self, image):
        #Convert the input image to HSV
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        #White color mask
        lower_threshold = np.uint8([0, 0, 150])
        upper_threshold = np.uint8([170, 135, 255])
        white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
        
        #Yellow color mask
        lower_threshold = np.uint8([11, 80, 80])
        upper_threshold = np.uint8([30, 255, 255])
        yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
        
        #Combine white and yellow masks
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked_image = cv2.bitwise_and(image, image, mask = mask)
        return masked_image

    def HLS_color_selection(self, image):
        #Convert the input image to HSL
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        #White color mask
        lower_threshold = np.uint8([0, 130, 0])
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

    def canny(self, image):
        kernel = np.ones((5,5), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        # canny = cv2.dilate(canny, kernel, iterations=1)
        # canny = cv2.erode(canny, kernel, iterations=1)
        return canny

    # .region selection
    def region_of_interest(self, image):
        arr = []
        mask = np.zeros_like(image)   
        height, width = image.shape[:2]
        point_1 = [width * 0.3, height * 0.88]
        point_2 = [width * 0.5, height * 0.69]
        point_3 = [width * 0.75, height * 0.69]
        point_4 = [width * 0.9, height * 0.88]
        arr.append([point_1, point_2, point_3, point_4])
        polygon = np.array(arr, dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image

    def pixel_points(self, image, line_parameters):
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.00001 ,0

        y1 = image.shape[0]
        y2 = int(y1*0.7)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept (self, image, lines):
        left_fit = []
        right_fit = []
        left_line = np.array([0, 0, 0, 0])
        right_line = np.array([0, 0, 0, 0])

        for line in lines:
            for x1, y1, x2, y2 in line:
                # slope = (y2 - y1) / (x2 - x1) # OverflowError: cannot convert float infinity to integer
                # intercept = y1 - (slope * x1)
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = (self.pixel_points(image, left_fit_average))
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = (self.pixel_points(image, right_fit_average))
        return np.array([left_line, right_line])

    def display_lines (self, image, lines):
        # display black
        line_image = np.zeros_like(image)
        h, w = image.shape[0], image.shape[1]
        temp = [0]
        arr = []
        two_lines = False
        one_line = False
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                if (0 < x1 < w  and 0 < x2 < w):
                    arr.append([x1, y1])
                    arr.append([x2, y2])
            if len(arr) == 4:
                temp = arr[2]; arr[2] = arr[3]; arr[3] = temp # doi vi tri cac diem da giac
                p1 = arr[0]; p2 = arr[1]; p3 = arr[2]; p4 = arr[3]
                if ((p2[0] < (p3[0] - int(w*0.1)))
                    and (p2[0] - p1[0] < int(w*0.5)) 
                    and (p4[0] - p3[0] < int(w*0.5))):
                    if (int(w*0.4) < p3[0] < int(w*0.8)) and  (int(w*0.2) < p2[0] < int(w*0.6)):
                        pts = np.array(arr, np.int32)
                        cv2.fillPoly(line_image, [pts], (0,255,0))
                        cv2.line(line_image, p1, p2, (0, 0, 255), 5)
                        cv2.line(line_image, p3, p4, (0, 0, 255), 5)
                        two_lines = True
            if len(arr) == 2:
                p1_x = arr[0][0]
                p2_x = arr[1][0]
                if ((p1_x < p2_x and p1_x < int(w*0.6) and p2_x < int(w*0.6))
                    or (p1_x > p2_x and p1_x > int(w*0.4) and p2_x > int(w*0.4))):
                    cv2.line(line_image, arr[0], arr[1], (0, 0, 255), 5)
                    one_line = True
                    
        return line_image, arr ,two_lines, one_line


    