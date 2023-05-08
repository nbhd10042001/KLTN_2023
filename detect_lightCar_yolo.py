import cv2
import time
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5

# Load vehicle detector
vd = VehicleDetector_yolov5()
# video = cv2.VideoCapture("video/car/car1.mp4")
video = cv2.VideoCapture("video\light_blink\light_blink6.mp4")
# video = cv2.VideoCapture("video/lane2.mp4")

while True:
    _, frame = video.read()
    frame = cv2.resize(frame, [640 ,480])
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameEqua = cv2.equalizeHist(frameGray)

    light_boxes, cls_b = vd.detect_LightVehicles(frameEqua)
    print(cls_b)

    if light_boxes:
        for box in light_boxes:
            x, y, w, h, conf = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(frame,"{:.2f}".format(conf), (x, y), 0, 0.5, (0, 0, 255), 1)
    
    cv2.imshow("Car", frame)
    # cv2.imshow("Car1", frameGray)
    # cv2.imshow("Car2", frameEqua)

    key = cv2.waitKey(1)
    if key == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break

