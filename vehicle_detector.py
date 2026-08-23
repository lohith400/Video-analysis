"""
vehicle_detector.py — YOLO-based vehicle and person detection
"""

from ultralytics import YOLO
import numpy as np

VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    1:  "bicycle",
}
PERSON_CLASS = 0


class VehicleDetector:
    def __init__(self, model_path: str, confidence: float = 0.5):
        print(f"  Loading vehicle model: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = confidence

    def _run(self, frame):
        return self.model(frame, conf=self.conf, verbose=False)[0]

    def detect_vehicles(self, frame):
        """Returns list of (box_xyxy_int, class_name)."""
        results = self._run(frame)
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in VEHICLE_CLASSES:
                coords = list(map(int, box.xyxy[0]))
                detections.append((coords, VEHICLE_CLASSES[cls]))
        return detections

    def detect_persons(self, frame):
        """Returns list of (box_xyxy_int, 'person')."""
        results = self._run(frame)
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == PERSON_CLASS:
                coords = list(map(int, box.xyxy[0]))
                detections.append((coords, "person"))
        return detections
