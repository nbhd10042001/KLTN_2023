import cv2
import time
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
import os

pathFile = os.path.dirname(__file__)
pathVideo = os.path.join(pathFile, "video")
# video = pathVideo + "/car/car3_Trim.mp4"
# video = pathVideo + "/light_blink/light_blink2.mp4"
video = pathVideo + "/lane4.mp4"

# Load vehicle detector
vd = VehicleDetector_yolov5()

cap = cv2.VideoCapture(video)
# cap = cv2.VideoCapture(0)

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(video)
        continue

    frame = cv2.resize(frame, [640 ,480])
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frameEquaHisto = cv2.equalizeHist(frameGray)
    vehicle_boxes, cls_b, lightsBox = vd.detect_vehicles(frame)
    print(cls_b)
    vehicle_count = len(vehicle_boxes)

    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, conf = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(frame,"{:.2f}".format(conf), (x, y), 0, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)
    
    if lightsBox:
        for box in lightsBox:
            x, y, w, h, conf = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,0), 2)

    cv2.imshow("Car", frame)
    cv2.imshow("equahis", frameEquaHisto)
    cv2.imshow("gray", frameGray)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    end = time.time()
    print("frame/s: "+"{:.3f}s".format(end - start))
cap.release()
cv2.destroyAllWindows()
