import numpy as np
import torch
import os

pathFile = os.path.dirname(__file__)
path1 = os.path.join(pathFile,"..\..\yolov5")
path2 = os.path.join(pathFile,"..\weights\yolov5s.pt")
# path2 = os.path.join(pathFile,"..\weights\carl300e.pt")

class VehicleDetector_yolov5:

    def __init__(self):
        # load model
        # weights_select  = input("select your weights: ")
        # model = torch.hub.load('','yolov5', 'weights\yolov5s', source='local')
        # self.model = torch.hub.load('..\yolov5','custom', 'weights\{}'.format(weights_select), source='local', device = 'cpu') # path theo terminal
        # self.model = torch.hub.load('..\yolov5','custom', 'weights\car_lights1.pt', source='local', device = 'cpu') # path theo terminal
        self.model = torch.hub.load(path1,'custom', path2, source='local', device = 'cpu') # path theo terminal

    def detect_vehicles(self, frame):
        vehicles_boxes = []
        lightsBox = []
        name_boxes = []
        class_name = ['person', 'bicycle', 'motorcycle']
        class_car = ['car', 'bus', 'truck']
        # detect
        detections = self.model(frame)
        results = detections.pandas().xyxy[0].to_dict(orient="records")

        if results:
            for result in results:
                name = result['name']
                class_ = result['class']
                confidence = result['confidence']

                for m in class_car:
                    if name == m and confidence > 0.3:
                        x1 = int(result['xmin'])
                        y1 = int(result['ymin'])
                        x2 = int(result['xmax'])
                        y2 = int(result['ymax'])
                        w = x2 - x1
                        h = y2 - y1
                        conf = confidence
                        box = [x1, y1, w, h, conf]
                        vehicles_boxes.append(box)

                for n in class_name:
                    if name == n and confidence > 0.3:
                        x1 = int(result['xmin'])
                        y1 = int(result['ymin'])
                        x2 = int(result['xmax'])
                        y2 = int(result['ymax'])
                        w = x2 - x1
                        h = y2 - y1
                        conf = confidence
                        box = [x1, y1, w, h, conf]
                        name_boxes.append(box)

                if (name == "light" and confidence > 0.5):
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    w = x2 - x1
                    h = y2 - y1
                    conf = confidence
                    box = [x1, y1, w, h, conf]
                    lightsBox.append(box)
    
        return vehicles_boxes, name_boxes, lightsBox
