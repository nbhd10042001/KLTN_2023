import cv2
import numpy as np

class LightSignal_and_Warnings:

    def lane_crossing_warning(self, frame, box, masked_image):
        x, y, w, h, conf = box
        cv2.putText(frame,"{:.2f}".format(conf), (x, y), 0, 0.5, (255, 255, 0), 1)
        center = [int(x + w/2 -1), int(y + h -1)] #center box of vehicle (-1 de khong bi IndexError: index 480 is out of bounds for axis 0 with size 480)
        cv2.circle(masked_image, center, 5, 255, 2)
        b,g,r = masked_image[center[1], center[0]]
        if (b == 0 and g == 0 and r == 255):
            cv2.putText(frame,"Warning! Co xe vuot lan", (10, 20), 0, 0.5, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame, masked_image

    def create_mask_hsv(self, crop_lights):
        hsv = cv2.cvtColor(crop_lights, cv2.COLOR_BGR2HSV)
        # find mask and threshold
        lower = np.uint8([11, 80, 80])
        upper = np.uint8([30, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        # mask = cv2.erode(mask, kernel, iterations=2)
        return mask
    
    def create_mask_hls(self, crop_lights):
        hls = cv2.cvtColor(crop_lights, cv2.COLOR_BGR2HLS)
        # find mask and threshold
        lower = np.uint8([10, 0, 100])
        upper = np.uint8([40, 255, 255])

        mask = cv2.inRange(hls, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        # mask = cv2.erode(mask, kernel, iterations=2)
        return mask
    
    def handle_lightSignal(self,frame, classCar, lightBoxs):
        for car in classCar:
            cx = car.x # x, y, w, h of car
            cy = car.y
            cw = car.w
            ch = car.h
            P_car = 2*(cw + ch) # caculate rectangular perimeter
            for lightBox in lightBoxs:
                xl, yl, wl, hl = lightBox
                xl_cent = xl + int(wl/2) # find center of box light
                yl_cent = yl + int(hl/2)
                if cx <= xl_cent <= (cx + cw) and cy <= yl_cent <= (cy + ch): # check light is exist in car
                    car.numberLight += 1
                    P_light = 2*(wl + hl)
                    if P_light > int(P_car*0.05):
                        # detect turn signal lights 
                        if (cx <= xl_cent <= (cx + int((cw)/3))) and ((cy) <= yl_cent <= (cy + ch)):
                            car.turnLeft = True
                            cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                            cv2.putText(frame, "Left", (xl, yl), 0, 0.3, (0, 255, 255), 1)

                        if ((cx + cw)- int((cw)/3)) <= xl_cent <= (cx + cw) and (cy) <= yl_cent <= (cy + ch):
                            car.turnRight = True
                            cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                            cv2.putText(frame, "Right", (xl, yl), 0, 0.3, (0, 255, 255), 1)
            if car.numberLight < 3:
                if car.turnRight == True and car.turnLeft == True:
                    cv2.putText(frame,"Warning! Xe dung khan cap!", (10, 40), 0, 0.5, (0, 255, 255), 2)
                if car.turnRight == True and car.turnLeft == False:
                    cv2.putText(frame,"Warning! Xe re", (10, 60), 0, 0.5, (0, 255, 255), 2)
                if car.turnRight == False and car.turnLeft == True:
                    cv2.putText(frame,"Warning! Xe re", (10, 60), 0, 0.5, (0, 255, 255), 2)
        return frame
    
    def crop_lights_vehicle(self, image, boxs):
        arr = []
        for i in range(len(boxs)):
            x, y, w, h, cf = boxs[i]
            # add box to arr
            h_4 = int(h/3)
            w_3 = int(w/3)
            # arr.append([(x, y + h2), (x + w_3, y + h2), (x + w_3, y+h), (x, y+h)])
            # arr.append([(x + w33, y + h2), (x + w, y + h2), (x + w, y+h), (x + w33, y+h)])
            arr.append([(x, y+h_4), (x + w_3, y+h_4), (x + w_3, y+h), (x, y+h)])
            arr.append([((x+w)-w_3, y+h_4), (x + w, y+h_4), (x + w, y+h), ((x+w)-w_3, y+h)])
        polygons = np.array(arr)
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, polygons, (255,255,255))
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
