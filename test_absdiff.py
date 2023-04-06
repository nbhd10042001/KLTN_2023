import cv2
import numpy as np

video = cv2.VideoCapture("./video/light_blink/light4.mp4")
last_f = None
while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture(0)
        continue

    frame_copy = frame.copy()

    # detect color lights
    hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)

    # find mask and threshold
    lower = np.array([17, 40, 180], dtype="uint8")
    upper = np.array([38, 255, 255], dtype="uint8")
    
    mask = cv2.inRange(hsv, lower, upper)
    bitw = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
    kernel = np.ones((5, 5), np.uint8)

    frame_copy = cv2.dilate(mask, kernel, iterations=2)
    # frame_copy = cv2.erode(mask, kernel, iterations=2)
    # frame_copy = cv2.dilate(mask, kernel, iterations=2)


    if last_f is None:
        last_f = frame_copy
        continue

    diff = cv2.absdiff(frame_copy, last_f)
    last_f = frame_copy

    contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop contuors-------------------------------------------------------------------
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        # loc nhieu
        if area > 100:
            print("co den nhap nhay")
            # cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x1 ,y1), (x1 + w1, y1 + h1), (0,0,255), 3)


    cv2.imshow("diff", diff)
    cv2.imshow("frame_copy", frame_copy)
    cv2.imshow("frame", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break