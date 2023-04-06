import cv2
import numpy as np

class LaneDetector:
    def make_coordinates(self, image, line_parameters):
        # slope, intercept = line_parameters
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.001 ,0

        y1 = image.shape[0]
        y2 = int(y1 - y1/3) 
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept (self, image, lines):
        left_fit = []
        right_fit = []
        left_line = np.array([0, 0, 0, 0])
        right_line = np.array([0, 0, 0, 0])

        for line in lines:
            x1,y1,x2,y2 =line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = abs(self.make_coordinates(image, left_fit_average))
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = abs(self.make_coordinates(image, right_fit_average))
        
        return np.array([left_line, right_line])

    def canny(self, image):
        kernel = np.ones((5,5), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ret, thresh = cv2.threshold(gray, 100,255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150, apertureSize=3)

        # result = cv2.bitwise_and(canny, thresh)
        # result = cv2.dilate(result, kernel, iterations=1)
        # canny = cv2.erode(canny, kernel, iterations=1)
        return canny  

    def display_lines (self, image, lines):
        # display black
        line_image = np.zeros_like(image)
        h, w = image.shape[0], image.shape[1]
        temp = [0]
        arr = []
        if lines is not None:
            line1 = lines[0]
            line2 = lines[1]
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            if (w > x1 > 0  and w > x2 > 0):
                x2 += 100
                y2 -= 100
                arr.append([x1, y1])
                arr.append([x2, y2])
            if (w > x3 > 0  and w > x4 > 0):
                x4 -= 100
                y4 -= 100
                arr.append([x3, y3])
                arr.append([x4, y4])
            if len(arr) == 4:
                if (arr[1][0] < arr[3][0])  and arr[1][0] < w/2 and arr[3][0] > w/2:
                    temp = arr[2]; arr[2] = arr[3]; arr[3] = temp
                    pts = np.array(arr, np.int32)
                    cv2.fillPoly(line_image, [pts], (0,255,0))
        return line_image

    # create mask polygons
    def region_of_interest(self, image):
        h, w = image.shape[0], image.shape[1]
        arr = []
        p1 = (int(0), h)
        p2 = (int(w/2 - (w/4)/4), int(h - h/2))
        p3 = (int(w/2 + (w/4)/4), int(h - h/2))
        p4 = (int(w), h)

        arr.append([p1, p2, p3, p4])
        polygons = np.array(arr)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
