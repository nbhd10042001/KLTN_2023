import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


def nothing(x):
    # any operation
    pass

cv2.namedWindow('FRAME')
cv2.createTrackbar("brightness", "FRAME", 0, 100, nothing)
cv2.createTrackbar("contrast", "FRAME", 0, 100, nothing)

while True:
    _, frame = cap.read()
    cv2.imshow("frame", frame)

    brightness = cv2.getTrackbarPos('brightness', 'FRAME')
    contrast = cv2.getTrackbarPos('contrast', 'FRAME')
    cap.set(10, brightness)
    cap.set(11, contrast)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
