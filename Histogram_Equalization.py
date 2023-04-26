import cv2
import time
import os

pathCurrent = os.path.dirname(__file__)
pathVideo = os.path.join(pathCurrent, 'video')
# vd = pathVideo + "\car\car3_Trim.mp4"
vd = pathVideo + "\lcl.mp4"

video = cv2.VideoCapture(vd)
lasttime = time.time()
fps = float(1/60)
while True:
    start = time.time()

    if start - lasttime > fps:
        ret, frame = video.read()
        if not ret:
            video = cv2.VideoCapture(vd)
            continue
        
        frame = cv2.resize(frame, [640, 480])
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.equalizeHist(frameGray)

        cv2.imshow("frameGray", frameGray)
        cv2.imshow("frameEqualize", frame1)
        cv2.imshow("frame", frame)

        lasttime = start

    key = cv2.waitKey(1)
    if key == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break
    