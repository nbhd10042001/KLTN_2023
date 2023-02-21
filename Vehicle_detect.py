import cv2
import numpy as np

class VehicleDetector:

    def __init__(self):
        #load network
        net = cv2.dnn.readNet("model_dnn/yolov3-tiny.weights", "model_dnn/yolov3-tiny.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale = 1/255)

        #Allow classes containing vehicle only
        # self.classes_allowed = [2, 3, 5, 6, 7]
        self.classes_allowed = [2]


    def detect_vehicles(self, img):

        #detect Objects
        vehicles_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold = 0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.3:
                #skip detecion with low confidence
                continue

            if class_id in self.classes_allowed:
                vehicles_boxes.append(box)

        return vehicles_boxes
