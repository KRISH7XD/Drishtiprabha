from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, weights_path="yolov8n.pt"):
        self.model = YOLO(weights_path)

    def detect(self, frame):
        """Detect objects in the frame and return (label, center_x) pairs."""
        results = self.model(frame, verbose=False)
        detected_objects = []

        for box in results[0].boxes:
            label = self.model.names[int(box.cls)]
            confidence = box.conf.item()
            if confidence > 0.5:  # Confidence threshold
                x, y, w, h = box.xywh[0].tolist()
                center_x = x / frame.shape[1]  # Normalized x position
                detected_objects.append((label, center_x))

        return detected_objects