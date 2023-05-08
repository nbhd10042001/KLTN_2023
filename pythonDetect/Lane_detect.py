import cv2
import numpy as np
import warnings    
warnings.simplefilter('ignore', np.RankWarning) 

class LaneDetector:
    def make_coordinates(self, image, line_parameters):
        # slope, intercept = line_parameters
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.0001 ,0

        y1 = image.shape[0]
        y2 = int(y1 - y1/3) #y1*(2/4)
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
            # print("======="+ parameters)
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
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # ret, thresh = cv2.threshold(blur, 55,255, cv2.THRESH_BINARY)
        canny = cv2.Canny(blur, 100, 200)

        # result = cv2.bitwise_and(canny, thresh)
        # result = cv2.dilate(result, kernel, iterations=1)
        canny = cv2.dilate(canny, kernel, iterations=1)
        # canny = cv2.erode(canny, kernel, iterations=1)
        return canny

    def display_lines (self, image, lines):
        # display black
        line_image = np.zeros_like(image)
        h, w = image.shape[0], image.shape[1]
        # print(h, w)
        temp = [0]
        arr = []
        centerLine = False
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                if (w > x1 > 0  and w > x2 > 0) and ((abs(x1) - abs(x2)) < w*0.3):
                    arr.append([x1, y1])
                    arr.append([x2, y2])
            if len(arr) == 4:
                temp = arr[2]; arr[2] = arr[3]; arr[3] = temp
                if (arr[1][0] < arr[2][0]-(w*0.02)):
                    if (int(w*0.4) < arr[2][0] < int(w*0.6)) and  (int(w*0.4) < arr[1][0] < int(w*0.6)):
                        pts = np.array(arr, np.int32)
                        cv2.fillPoly(line_image, [pts], (0,255,0))
                        cv2.line(line_image, arr[0], arr[1], (0, 0, 255), 5)
                        cv2.line(line_image, arr[2], arr[3], (0, 0, 255), 5)
                        centerLine = True
        return line_image, arr ,centerLine

    # create mask polygons
    def region_of_interest(self, image):
        h, w = image.shape[0], image.shape[1]
        arr = []
        p1 = (0*w, h)
        p2 = (int(w*0.45), int(h*0.7))
        p3 = (int(w*0.55), int(h*0.7))
        p4 = (w, h)

        arr.append([p1, p2, p3, p4])
        polygons = np.array(arr)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    #perspective transform on undistorted images
    def perspective_transform(self, img):
        imshape = img.shape
        #print (imshape)
        vertices = np.array([[(.55*imshape[1], 0.63*imshape[0]), (imshape[1],imshape[0]),
                        (0,imshape[0]),(.45*imshape[1], 0.63*imshape[0])]], dtype=np.float32)
        #print (vertices)
        src= np.float32(vertices)
        dst = np.float32([[0.75*img.shape[1],0],[0.75*img.shape[1],img.shape[0]],
                        [0.25*img.shape[1],img.shape[0]],[0.25*img.shape[1],0]])
        #print (dst)
        M = cv2.getPerspectiveTransform(src, dst)

        Minv = cv2.getPerspectiveTransform(dst, src)
        img_size = (imshape[1], imshape[0]) 
        perspective_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)    
        return perspective_img, Minv
    