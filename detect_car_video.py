import cv2
from Vehicle_detect import VehicleDetector

# Load vehicle detector
vd = VehicleDetector()

video = cv2.VideoCapture("video/test2.mp4")
# video = cv2.VideoCapture(0)

# Loop through the images
while True:
    _, frame = video.read()

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

video.release()
cv2.destroyAllWindows()
