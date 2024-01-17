import cv2
from ultralytics import YOLO

import matplotlib
# matplotlib.use('TkAgg')

class DetectionModel():
    def __init__(self, model_path = './models/yolov8n.pt'):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def pred(self, image):
        results = self.model(image, save=False)
        return results, self.class_names

