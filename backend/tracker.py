"""ByteTrack multi-object tracking via Ultralytics model.track()."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

import config
from schemas import TrackedVehicle  # re-exported for backward compatibility

__all__ = ["TrackedVehicle", "VehicleTracker"]


class MergedBoxes:
    """Combines vehicle and person detections while preserving downstream interface (.xyxy, .cls, .id, .conf)."""

    def __init__(self, xyxy, cls, id, conf):
        self.xyxy = xyxy
        self.cls = cls
        self.id = id
        self.conf = conf

    def __len__(self) -> int:
        return len(self.xyxy) if self.xyxy is not None else 0


class VehicleTracker:
    """Wraps YOLOv8/YOLO11 + ByteTrack for persistent vehicle tracking."""

    def __init__(
        self, model: YOLO, device: str, vehicle_class_ids: Optional[List[int]] = None
    ):
        self.model = model
        self.device = device
        self._vehicle_class_ids = vehicle_class_ids or []
        self._class_resolver = _build_class_resolver(model.names)
        self.last_boxes = None

        # Check if pedestrian (class 0 = 'person') needs an auxiliary tracker
        self._person_model = None
        has_person_class = any(str(name).lower() == "person" for name in model.names.values())
        if not has_person_class:
            person_path = Path("models/yolov8n.pt")
            if not person_path.exists():
                alt = Path("backend/models/yolov8n.pt")
                if alt.exists():
                    person_path = alt
                else:
                    alt2 = Path("../models/yolov8n.pt")
                    if alt2.exists():
                        person_path = alt2
            if person_path.exists():
                try:
                    self._person_model = YOLO(str(person_path))
                    self._person_model.to(device)
                except Exception as e:
                    print(f"[VehicleTracker] Notice: Could not init auxiliary pedestrian tracker: {e}")

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
            self.last_boxes = None
            return []

        boxes = results[0].boxes
        
        # Track pedestrians for downstream pedestrian detection if model lacks person class
        if self._person_model is not None:
            try:
                p_results = self._person_model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    tracker=config.TRACKER_CONFIG,
                    conf=config.CONF_THRESHOLD,
                    iou=config.IOU_THRESHOLD,
                    half=config.USE_HALF,
                    device=self.device,
                    verbose=False,
                )
                p_boxes = p_results[0].boxes if (p_results and p_results[0].boxes is not None) else None
            except Exception:
                p_boxes = None
            self.last_boxes = self._merge_boxes(boxes, p_boxes)
        else:
            self.last_boxes = boxes

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

    def _merge_boxes(self, v_boxes, p_boxes):
        if v_boxes is None and p_boxes is None:
            return None

        target_device = self.device

        # Shift vehicle class IDs so they never collide with person (cls 0)
        v_xyxy = v_boxes.xyxy.to(target_device) if (v_boxes is not None and v_boxes.xyxy is not None) else torch.empty((0, 4), device=target_device)
        v_cls = (v_boxes.cls.clone().to(target_device) + 10) if (v_boxes is not None and v_boxes.cls is not None) else torch.empty((0,), device=target_device)
        v_id = v_boxes.id.to(target_device) if (v_boxes is not None and v_boxes.id is not None) else None
        v_conf = v_boxes.conf.to(target_device) if (v_boxes is not None and v_boxes.conf is not None) else None

        if p_boxes is None or p_boxes.id is None or len(p_boxes) == 0:
            return MergedBoxes(v_xyxy, v_cls, v_id, v_conf)

        p_xyxy = p_boxes.xyxy.to(target_device)
        p_cls = torch.zeros((len(p_boxes),), dtype=v_cls.dtype, device=target_device)  # 0 is person
        p_id = (p_boxes.id.to(target_device) + 100000) if p_boxes.id is not None else None
        p_conf = p_boxes.conf.to(target_device) if p_boxes.conf is not None else None

        if v_id is not None and p_id is not None:
            merged_xyxy = torch.cat([v_xyxy, p_xyxy], dim=0)
            merged_cls = torch.cat([v_cls, p_cls], dim=0)
            merged_id = torch.cat([v_id, p_id], dim=0)
            merged_conf = torch.cat([v_conf, p_conf], dim=0) if (v_conf is not None and p_conf is not None) else None
            return MergedBoxes(merged_xyxy, merged_cls, merged_id, merged_conf)
        elif p_id is not None:
            return MergedBoxes(p_xyxy, p_cls, p_id, p_conf)
        else:
            return MergedBoxes(v_xyxy, v_cls, v_id, v_conf)


def _build_class_resolver(model_names: dict):
    """Map YOLO class id/name to one of ALL_VEHICLE_CLASSES."""
    id_to_name: dict[int, Optional[str]] = {}
    indian_map = getattr(config, "INDIAN_VEHICLE_CLASS_MAP", {})

    for cls_id, raw_name in model_names.items():
        cid = int(cls_id)
        name = str(raw_name).lower().replace("_", "-").strip()
        if name in indian_map:
            id_to_name[cid] = indian_map[name]
        elif name in config.ALL_VEHICLE_CLASSES:
            id_to_name[cid] = name
        elif cid in config.COCO_VEHICLE_ID_MAP:
            id_to_name[cid] = config.COCO_VEHICLE_ID_MAP[cid]

    def resolve(cls_id: int) -> Optional[str]:
        return id_to_name.get(cls_id)

    return resolve
