import cv2
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
import matplotlib.pyplot as plt
import glob  
import time


# Load vehicle detector
vd = VehicleDetector_yolov5()

# Load images from a folder
images_folder = glob.glob("img/car/*.*") # muon tim cu the thi dung ("img/*.jpg")
# print(images_folder)


# Loop through the images
for img_path in images_folder:
    # print("img_path:", img_path)
    
    start = time.time()

    img = cv2.imread(img_path)
    img = cv2.resize(img, [640, 480])

    vehicle_boxes, cls_b, plateBoxes = vd.detect_vehicles(img)
    # print ("box car:",vehicle_boxes)
    vehicle_count = len(vehicle_boxes)

    if vehicle_boxes:
        for box in vehicle_boxes:
            x, y, w, h, cf = box

            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)
            cv2.putText(img, "{:.2f}".format(cf), (x, y), 0, 0.5, (0, 0, 255), 2)

            # h_i = img.shape[0]
            # #calculate distance
            # dis = ((h_i) / h)
            # cv2.putText(img, "Dis: " + str(round(dis, 2)) + "m", (x, y), 0, 1, (255, 0, 0), 2)
    if plateBoxes:
        for box in plateBoxes:
            x, y, w, h, conf = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

    end = time.time()
    print("frame/second: ", end - start)
    cv2.imshow("Car", img)
    cv2.waitKey(0)

    # plt.imshow(img)
    # plt.show()
