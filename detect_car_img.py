import cv2
from pythonDetect.Vehicle_detect import VehicleDetector
from pythonDetect.Detect_yolov5 import VehicleDetector_yolov5
import matplotlib.pyplot as plt
import glob  


# Load vehicle detector
vd = VehicleDetector_yolov5()

# Load images from a folder
images_folder = glob.glob("img/car/*.*") # muon tim cu the thi dung ("img/*.jpg")
# print(images_folder)


# Loop through the images
for img_path in images_folder:
    # print("img_path:", img_path)

    img = cv2.imread(img_path)
    img = cv2.resize(img, [960, 640])

    vehicle_boxes = vd.detect_vehicles(img)
    print ("box car:",vehicle_boxes)
    vehicle_count = len(vehicle_boxes)


    for box in vehicle_boxes:
        x, y, w, h = box

        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (0, 255, 0), 2)

        h_i = img.shape[0]
        #calculate distance
        dis = ((h_i) / h)
        cv2.putText(img, "Dis: " + str(round(dis, 2)) + "m", (x, y), 0, 1, (255, 0, 0), 2)

 
    cv2.imshow("Car", img)
    cv2.waitKey(0)

    # plt.imshow(img)
    # plt.show()
