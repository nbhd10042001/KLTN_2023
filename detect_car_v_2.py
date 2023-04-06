import cv2
import numpy as np
from PIL import ImageGrab
import time
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5


# Load vehicle detector
vd = VehicleDetector_yolov5()

video = cv2.VideoCapture("video/road_car.mp4")
# video = cv2.VideoCapture(0)

last_time = time.time()
# Loop through the images
while True:
    # ret, frame = video.read()
    # frame = cv2.resize(frame, [1280,720])

    frame = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = frame.shape[0], frame.shape[1]
    frame = frame[0:int(h-(h/4)),0:w] # [y1:y2, x1:x2]

    vehicle_boxes, _ = vd.detect_vehicles(frame)

    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, cf = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

    cv2.imshow("Car", frame)

    print("time: "+"{:.4f}s".format(time.time() - last_time))
    last_time = time.time()

    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break
    


