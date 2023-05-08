import cv2
import numpy as np

class LightSignal_and_Warnings:

    def lane_crossing_warning(self, frame, box, masked_image):
        x, y, w, h, conf = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(frame,"{:.2f}".format(conf), (x, y), 0, 0.5, (255, 255, 0), 1)
        center = [int(x + w/2 -1), int(y + h -1)] #center box of vehicle (-1 de khong bi IndexError: index 480 is out of bounds for axis 0 with size 480)
        cv2.circle(masked_image, center, 5, 255, 2)
        b,g,r = masked_image[center[1], center[0]]
        if (b == 0 and g == 0 and r == 255):
            cv2.putText(frame,"Warning! Co xe vuot lan", (20, 110), 0, 1, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame, masked_image

    def create_mask_hsv(self, crop_lights):
        hsv = cv2.cvtColor(crop_lights, cv2.COLOR_BGR2HSV)
        # find mask and threshold
        lower = np.array([14, 140, 140])
        upper = np.array([179, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        # mask = cv2.erode(mask, kernel, iterations=2)
        return mask
    
    def handle_lightSignal(self,frame, classCar, lightBoxs):
        for car in classCar:
            for lightBox in lightBoxs:
                xl, yl, wl, hl = lightBox
                x_cent = xl + int(wl/2)
                y_cent = yl + int(hl/2)

                # detect turn signal lights 
                if (car.x <= x_cent <= (car.x + int((car.w)/3))) and ((car.y) <= y_cent <= (car.y + car.h)):
                    car.turnLeft = True
                    cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                    cv2.putText(frame, "Left", (xl, yl), 0, 0.5, (0, 255, 255), 2)

                if ((car.x + car.w)- int((car.w)/3)) <= x_cent <= (car.x + car.w) and (car.y) <= y_cent <= (car.y + car.h):
                    car.turnRight = True
                    cv2.rectangle(frame, (xl ,yl), (xl + wl, yl + hl), (0,255,255), 1)
                    cv2.putText(frame, "Right", (xl, yl), 0, 0.5, (0, 255, 255), 2)
    
            if car.turnRight == True and car.turnLeft == True:
                cv2.putText(frame,"Warning! Xe dung khan cap!", (20, 80), 0, 1, (0, 255, 255), 2)
            if car.turnRight == True and car.turnLeft == False:
                cv2.putText(frame,"Warning! Xe re phai", (20, 80), 0, 1, (0, 255, 255), 2)
            if car.turnRight == False and car.turnLeft == True:
                cv2.putText(frame,"Warning! Xe re trai", (20, 80), 0, 1, (0, 255, 255), 2)
        return frame
    
    def crop_lights_vehicle(self, image, boxs):
        arr = []
        for i in range(len(boxs)):
            x, y, w, h, cf = boxs[i]
            # add box to arr
            h14 = int(h/4)
            w13 = int(w/3)
            # arr.append([(x, y + h2), (x + w13, y + h2), (x + w13, y+h), (x, y+h)])
            # arr.append([(x + w33, y + h2), (x + w, y + h2), (x + w, y+h), (x + w33, y+h)])
            arr.append([(x, y+h14), (x + w13, y+h14), (x + w13, y+h), (x, y+h)])
            arr.append([((x+w)-w13, y+h14), (x + w, y+h14), (x + w, y+h), ((x+w)-w13, y+h)])
        polygons = np.array(arr)
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, polygons, (255,255,255))
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
