import cv2
import time
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
# from pythonDetect.Vehicle_detect import VehicleDetector

# Load vehicle detector
vd = VehicleDetector_yolov5()

video = cv2.VideoCapture("video/car1.mp4")
# video = cv2.VideoCapture(0)

# Loop through the images
while True:
    start = time.time()
    _, frame = video.read()
    frame = cv2.resize(frame, [1280 ,720])

    vehicle_boxes, cls_b = vd.detect_vehicles(frame)

    # print ("dt: ",vehicle_boxes)
    vehicle_count = len(vehicle_boxes)
    # print(cls_b)

    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, conf = box

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(frame,"{:.2f}".format(conf), (x, y), 0, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)
    
    cv2.imshow("Car", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    end = time.time()
    print("frame/s: "+"{:.3f}s".format(end - start))
video.release()
cv2.destroyAllWindows()
