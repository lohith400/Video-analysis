"""ByteTrack multi-object tracking via Ultralytics model.track()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: tuple  # x1, y1, x2, y2
    confidence: float
    vehicle_class: str


class VehicleTracker:
    """Wraps YOLOv8 + ByteTrack for persistent vehicle tracking."""

    def __init__(
        self, model: YOLO, device: str, vehicle_class_ids: Optional[List[int]] = None
    ):
        self.model = model
        self.device = device
        self._vehicle_class_ids = vehicle_class_ids or []
        self._class_resolver = _build_class_resolver(model.names)

    def track(self, frame: np.ndarray) -> List[TrackedVehicle]:
        kwargs = dict(
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            half=config.USE_HALF,
            device=self.device,
            verbose=False,
        )
        if self._vehicle_class_ids:
            kwargs["classes"] = self._vehicle_class_ids

        results = self.model.track(frame, **kwargs)
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        if boxes.id is None:
            return []

        vehicles: List[TrackedVehicle] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            vehicle_class = self._class_resolver(int(cls_ids[i]))
            if vehicle_class is None:
                continue
            x1, y1, x2, y2 = xyxy[i]
            vehicles.append(
                TrackedVehicle(
                    track_id=int(track_ids[i]),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(confs[i]),
                    vehicle_class=vehicle_class,
                )
            )
        return vehicles


def _build_class_resolver(model_names: dict):
    """Map YOLO class id/name to one of ALL_VEHICLE_CLASSES."""
    id_to_name: dict[int, Optional[str]] = {}

    for cls_id, raw_name in model_names.items():
        cid = int(cls_id)
        name = str(raw_name).lower().replace("_", "-")
        if name in config.ALL_VEHICLE_CLASSES:
            id_to_name[cid] = name
        elif cid in config.COCO_VEHICLE_ID_MAP:
            id_to_name[cid] = config.COCO_VEHICLE_ID_MAP[cid]

    def resolve(cls_id: int) -> Optional[str]:
        return id_to_name.get(cls_id)

    return resolve
