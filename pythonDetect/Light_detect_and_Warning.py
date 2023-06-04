import cv2
import numpy as np

class LightSignal_and_Warnings:

    def lane_crossing_warning(self, frame, box, masked_image):
        warning_cross = False
        x, y, w, h, conf, name = box
        # cv2.putText(frame,"{:.2f}".format(conf), (x, y+15), 0, 0.5, (255, 255, 0), 1)
        center = [int(x + w/2 -1), int(y + h -1)] #center box of vehicle (-1 de khong bi IndexError: index 480 is out of bounds for axis 0 with size 480)
        cv2.circle(masked_image, center, 5, 255, 2)
        b,g,r = masked_image[center[1], center[0]]
        if (b == 0 and g == 0 and r == 255):
            cv2.putText(frame,"                                                         ", (10, 50), 0, 0.5, (0, 0, 0), 2)
            cv2.putText(frame,"Warning! The {} crossing the lane".format(name), (10, 50), 0, 0.5, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            warning_cross = True
        return frame, masked_image, warning_cross

    def create_mask_hsv(self, crop_lights):
        hsv = cv2.cvtColor(crop_lights, cv2.COLOR_BGR2HSV)
        # find mask and threshold
        lower = np.array([17, 20, 255], dtype="uint8")
        upper = np.array([50, 255, 255], dtype="uint8")

        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # mask = cv2.erode(mask, kernel, iterations=2)
        return mask
    
    def handle_lightSignal(self,frame, classCar, lightBoxs):
        warning_signal = False
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
                    if P_light > int(P_car*0.03):
                        # detect turn signal lights 
                        if (cx <= xl_cent <= (cx + int((cw)/2))) and ((cy) <= yl_cent <= (cy + ch)):
                            car.turnLeft = True
                            cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                            cv2.putText(frame, "Left", (xl, yl), 0, 0.3, (0, 255, 255), 1)

                        if ((cx + cw)- int((cw)/2)) <= xl_cent <= (cx + cw) and (cy) <= yl_cent <= (cy + ch):
                            car.turnRight = True
                            cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                            cv2.putText(frame, "Right", (xl, yl), 0, 0.3, (0, 255, 255), 1)

            if car.numberLight < 4:
                if car.turnRight == True and car.turnLeft == True:
                    cv2.putText(frame,"                                                ", (10, 90), 0, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame,"Warning! The {} emergency stop!".format(car.name), (10, 90), 0, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
                if (car.turnRight == True and car.turnLeft == False 
                    or car.turnRight == False and car.turnLeft == True):
                    cv2.putText(frame,"                                                         ", (10, 70), 0, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame,"Warning! The {} wants to cross the lane!".format(car.name), (10, 70), 0, 0.5, (0, 255, 255), 2)
                    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 2)
                    warning_signal = True
        return frame, warning_signal
    
    def crop_lights_vehicle(self, image, boxs):
        arr = []
        for i in range(len(boxs)):
            x, y, w, h, _, _ = boxs[i]
            # add box to arr
            h_3 = int(h/3)
            w_3 = int(w*0.45)
            # arr.append([(x, y + h2), (x + w_3, y + h2), (x + w_3, y+h), (x, y+h)])
            # arr.append([(x + w33, y + h2), (x + w, y + h2), (x + w, y+h), (x + w33, y+h)])
            arr.append([(x, y+h_3), (x + w_3, y+h_3), (x + w_3, y+h), (x, y+h)])
            arr.append([((x+w)-w_3, y+h_3), (x + w, y+h_3), (x + w, y+h), ((x+w)-w_3, y+h)])
        polygons = np.array(arr)
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, polygons, (255,255,255))
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
