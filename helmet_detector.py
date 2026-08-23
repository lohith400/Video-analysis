import cv2
from ultralytics import YOLO

class HelmetDetector:
    def __init__(self, model_path="helmet_model.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_helmet(self, frame, person_box):
        """
        person_box: (x1, y1, x2, y2)
        returns: 'helmet', 'no_helmet', 'unknown'
        """

        x1, y1, x2, y2 = map(int, person_box)

        # Crop only upper body (head focus)
        h = y2 - y1
        head_y2 = y1 + int(h * 0.5)

        head_crop = frame[y1:head_y2, x1:x2]

        if head_crop is None or head_crop.size == 0:
            return "unknown"

        # Run helmet model
        results = self.model(head_crop, conf=self.conf, verbose=False)[0]

        if not results.boxes:
            return "unknown"

        # Pick best detection
        best = max(results.boxes, key=lambda b: float(b.conf[0]))
        cls_id = int(best.cls[0])
        label = results.names[cls_id].lower()

        # Robust classification
        if label in ["no_helmet", "no helmet", "without helmet"]:
            return "no_helmet"
        elif label in ["helmet", "with helmet"]:
            return "helmet"

        return "unknown"