import cv2
import numpy as np
import time

# video = cv2.VideoCapture("video\slow_traffic_small.mp4")
# video = cv2.VideoCapture("video\car_light3_Trim.mp4")
video = cv2.VideoCapture("video/lane/ok3.mp4")
# video = cv2.VideoCapture("video/lane3.mp4")

video.set(10, 0)
video.set(11, 0)
font = cv2.FONT_HERSHEY_COMPLEX

#create trackbar
def nothing(x):
    # any operation
    pass

cv2.namedWindow('FRAME')
cv2.createTrackbar("L-H", "FRAME", 0, 255, nothing)
cv2.createTrackbar("L-S", "FRAME", 150, 255, nothing)
cv2.createTrackbar("L-V", "FRAME", 0, 255, nothing)
cv2.createTrackbar("U-H", "FRAME", 255, 255, nothing)
cv2.createTrackbar("U-S", "FRAME", 255, 255, nothing)
cv2.createTrackbar("U-V", "FRAME", 255, 255, nothing)

start = time.time()
FPS = 30
while True:
    last = time.time()
    if last - start >  (1/FPS):
        start = last
        #load img
        _, frame = video.read(0)
        # frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=10)
        frame = cv2.resize(frame, [640, 480])

        # detect color lights
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        l_h = cv2.getTrackbarPos('L-H', 'FRAME')
        l_s = cv2.getTrackbarPos('L-S', 'FRAME')
        l_v = cv2.getTrackbarPos('L-V', 'FRAME')
        u_h = cv2.getTrackbarPos('U-H', 'FRAME')
        u_s = cv2.getTrackbarPos('U-S', 'FRAME')
        u_v = cv2.getTrackbarPos('U-V', 'FRAME')
        # find mask and threshold
        lower = np.array([l_h, l_s, l_v], dtype="uint8")
        upper = np.array([u_h, u_s, u_v], dtype="uint8")
        
        mask = cv2.inRange(hsv, lower, upper)
        bitw = cv2.bitwise_and(frame, frame, mask=mask)
        kernel = np.ones((5, 5), np.uint8)

        # mask = cv2.dilate(mask, kernel, iterations=3)
        # mask = cv2.erode(mask, kernel, iterations=2)
        # mask = cv2.dilate(mask, kernel, iterations=8)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop contuors-------------------------------------------------------------------
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            # loc nhieu
            if area > 50:
                # cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                x1, y1, w1, h1 = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x1 ,y1), (x1 + w1, y1 + h1), (0,255,0), 2)


        # if find contours (True)-------------------------------------------------------
        # if contours:
        #     # Find the index of the largest contour
        #     areas = [cv2.contourArea(c) for c in contours]
        #     max_index = np.argmax(areas)
        #     cnt=contours[max_index]

        #     # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        #     # cv2.drawContours(img_copy1, [approx], 0, (0, 0, 0), 5)
        #     x1, y1, w1, h1 = cv2.boundingRect(cnt)
        #     # print(x1, y1, w1, h1)
        #     cv2.rectangle(frame, (x1 ,y1), (x1 + w1, y1 + h1), (0,255,0), 2)
            
        
        cv2.imshow("mask_crop", mask)
        # cv2.imshow("bitw", bitw)
        cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()