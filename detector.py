"""Two-stage YOLOv8: vehicle tracking only.
PlateDetector is commented out — re-enable when license_plate_detector.pt is available."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class PlateDetector:
    """Runs license_plate_detector.pt on a vehicle crop (worker thread safe)."""

    def __init__(self, device: str):
        plate_path = Path(config.PLATE_MODEL)
        if not plate_path.exists():
            raise FileNotFoundError(
                f"Plate model not found: {plate_path}. "
                "Place license_plate_detector.pt in models/"
            )
        self.device = device
        self.model = YOLO(str(plate_path))
        self.model.to(device)

    def detect_plate_region(
        self, vehicle_crop: np.ndarray
    ) -> Optional[np.ndarray]:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        results = self.model.predict(
            vehicle_crop,
            conf=config.PLATE_CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            half=config.USE_HALF,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        h, w = vehicle_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return vehicle_crop[y1:y2, x1:x2].copy()

    def detect_plate_bbox(
        self, vehicle_crop: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """Returns the plate bounding box coordinates (x1, y1, x2, y2) and the crop image."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        results = self.model.predict(
            vehicle_crop,
            conf=config.PLATE_CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            half=config.USE_HALF,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        h, w = vehicle_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2), vehicle_crop[y1:y2, x1:x2].copy()


def crop_vehicle(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2].copy()


class VehicleModelLoader:
    """Loads vehicle YOLO model and filters to vehicle-related classes only."""

    def __init__(self, device: str):
        self.device = device
        model_path = Path(config.VEHICLE_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vehicle model not found: {model_path}. "
                "Run: python download_models.py"
            )
        self.model = YOLO(str(model_path))
        self.model.to(device)
        self._vehicle_class_ids = self._resolve_vehicle_class_ids()
        print(f"[VehicleModelLoader] Loaded {model_path} on {device}")

    def _resolve_vehicle_class_ids(self) -> List[int]:
        ids: List[int] = []
        for cls_id, raw_name in self.model.names.items():
            cid = int(cls_id)
            name = str(raw_name).lower().replace("_", "-")
            if name in config.ALL_VEHICLE_CLASSES:
                ids.append(cid)
            elif cid in config.COCO_VEHICLE_ID_MAP:
                ids.append(cid)
        # Include person (class 0) for pedestrian tracking
        if 0 not in ids:
            ids.append(0)
        return sorted(set(ids))

    @property
    def yolo(self) -> YOLO:
        return self.model

    @property
    def vehicle_class_ids(self) -> List[int]:
        return self._vehicle_class_ids