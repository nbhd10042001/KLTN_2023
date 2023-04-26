import numpy as np
import torch

class VehicleDetector_yolov5:

    def __init__(self):
        # load model
        # weights_select  = input("select your weights: ")
        # model = torch.hub.load('','yolov5', 'weights\yolov5s', source='local')
        # self.model = torch.hub.load('..\yolov5','custom', 'weights\{}'.format(weights_select), source='local', device = 'cpu') # path theo terminal
        self.model = torch.hub.load('..\yolov5','custom', 'weights\lightCar.pt', source='local', device = 'cpu') # path theo terminal

    def detect_vehicles(self, frame):
        vehicles_boxes = []
        plateBoxes = []
        class_box = []
        # detect
        detections = self.model(frame)
        results = detections.pandas().xyxy[0].to_dict(orient="records")

        if results:
            for result in results:
                name = result['name']
                class_ = result['class']
                confidence = result['confidence']

                class_box.append(name)
                class_box = list(set(class_box))

                if (class_ == 0 and confidence > 0.7) or (class_ == 2 and confidence > 0.5):
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    w = x2 - x1
                    h = y2 - y1
                    conf = confidence
                    box = [x1, y1, w, h, conf]
                    vehicles_boxes.append(box)

                if (class_ == 1 and confidence > 0.1):
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    w = x2 - x1
                    h = y2 - y1
                    conf = confidence
                    box = [x1, y1, w, h, conf]
                    plateBoxes.append(box)
    
        return vehicles_boxes, class_box, plateBoxes

    def detect_LightVehicles(self, frame):
        Light_boxes = []
        class_box = []
        # detect
        detections = self.model(frame)
        results = detections.pandas().xyxy[0].to_dict(orient="records")
        
        if results:
            for result in results:
                name = result['name']
                class_ = result['class']
                confidence = result['confidence']

                class_box.append(name)
                class_box = list(set(class_box))

                if (class_ == 0 and confidence > 0.5):
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    w = x2 - x1
                    h = y2 - y1
                    conf = confidence
                    box = [x1, y1, w, h, conf]
                    Light_boxes.append(box)
        return Light_boxes, class_box