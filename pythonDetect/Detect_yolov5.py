import numpy as np
import torch

class VehicleDetector_yolov5:

    def __init__(self):
        # load model
        # model = torch.hub.load('','yolov5', 'weights\yolov5s', source='local')
        self.model = torch.hub.load('..\yolov5','custom', 'weights\yolov5s.pt', source='local') # path theo terminal

        #Allow classes containing vehicle only
        # self.classes_allowed = [2, 3, 5, 6, 7]
        self.classes_allowed = [2]


    def detect_vehicles(self, frame):
        vehicles_boxes = []
        conf = []
        # detect
        detections = self.model(frame)
        results = detections.pandas().xyxy[0].to_dict(orient="records")

        if results:
            for result in results:
                name = result['name']
                class_ = result['class']
                confidence = result['confidence']

                if class_ == 2 and confidence > 0.5:
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    w = x2 - x1
                    h = y2 - y1
                    box = [x1, y1, w, h]
                    conf = [confidence]
                    vehicles_boxes.append(box)
                    
        return vehicles_boxes, conf
