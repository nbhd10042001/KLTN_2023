import cv2
import time
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
# from pythonDetect.Vehicle_detect import VehicleDetector
from tracker import *

# Load vehicle detector
vd = VehicleDetector_yolov5()

# Tao doi tuong tracking
tracker = EuclideanDistTracker()
vehicle_boxes = []

video = cv2.VideoCapture("video/road_car.mp4")
# video = cv2.VideoCapture(0)

_, frame = video.read()
vehicle_boxes, _, _ = vd.detect_vehicles(frame)
count = 0
# Loop through the images
while True:
    start = time.time()
    ret, frame = video.read()
    count += 1
    print(count)
    frame = cv2.resize(frame, [1280,720])

    if vehicle_boxes == []:
        vehicle_boxes, _, _ = vd.detect_vehicles(frame)

    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h = box
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

    # tracking objects--------------------------------------------------------------------------
    boxes_ids = tracker.update(vehicle_boxes)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), 0, 2, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

    if count == 10:
        count = 0
        vehicle_boxes = []
    
    cv2.imshow("Car", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    end = time.time()
    print("frame/s: "+"{:.3f}s".format(end - start))
video.release()
cv2.destroyAllWindows()
