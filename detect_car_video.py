import cv2
import time
from pythonDetect.Vehicle_detect import VehicleDetector
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5


# Load vehicle detector
vd = VehicleDetector_yolov5()

# video = cv2.VideoCapture("video/road_car.mp4")
video = cv2.VideoCapture(0)

# Loop through the images
while True:
    _, frame = video.read()
    start = time.time()

    vehicle_boxes = vd.detect_vehicles(frame)
    # print (vehicle_boxes)
    vehicle_count = len(vehicle_boxes)

    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(frame, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)

    cv2.imshow("Car", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    end = time.time()
    print("frame/s:", end - start)
video.release()
cv2.destroyAllWindows()
