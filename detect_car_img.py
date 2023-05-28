import cv2
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
import matplotlib.pyplot as plt
import glob  
import time


# Load vehicle detector
vd = VehicleDetector_yolov5()

# Load images from a folder
images_folder = glob.glob("img/speed_car/*.*") # muon tim cu the thi dung ("img/*.jpg")
# print(images_folder)


# Loop through the images
for img_path in images_folder:
    # print("img_path:", img_path)
    start = time.time()
    img = cv2.imread(img_path)
    img = cv2.resize(img, [640, 480])

    vehicle_boxes, bike_boxes = vd.detect_vehicles(img)
    # print ("box car:",vehicle_boxes)
    vehicle_count = len(vehicle_boxes) + len(bike_boxes)

    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, cf = box

            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, "{:.2f}".format(cf), (x, y), 0, 0.3, (0, 255, 0), 1)

    if bike_boxes:
        for box in bike_boxes:
            x, y, w, h, cf = box

            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, "{:.2f}".format(cf), (x, y), 0, 0.3, (0, 255, 0), 1)

    cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 1, (0, 255, 0), 2)
    end = time.time()
    cv2.imshow("Car", img)
    cv2.waitKey(0)

    # plt.imshow(img)
    # plt.show()
