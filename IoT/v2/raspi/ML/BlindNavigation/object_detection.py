from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detected_objects = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2 / frame.shape[1]
                detected_objects.append((label, x_center))
        
        return detected_objects