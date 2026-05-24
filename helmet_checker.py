"""Helmet Checker Vision Engine.
Runs models/helmet_detector.pt asynchronously inside a thread pool to detect helmet violations."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

import config


class HelmetChecker:
    def __init__(self, device: str):
        self.device = device
        self._lock = threading.Lock()

        helmet_path = Path(config.HELMET_MODEL)
        if not helmet_path.exists():
            print(f"[HelmetChecker] WARNING: Helmet model not found at {helmet_path}. Helmet detection will be disabled.")
            self.model = None
        else:
            self.model = YOLO(str(helmet_path))
            self.model.to(device)
            print(f"[HelmetChecker] Loaded helmet model on {device}")

        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # State tracking
        self.active_violations: Dict[int, List[Dict]] = {}  # track_id -> list of violations
        self.all_violations: List[Dict] = []               # All violations detected in this session
        self.attempts: Dict[int, int] = {}                  # track_id -> attempts count
        self.pending_futures: Dict[int, Future[Optional[List[Dict]]]] = {} # track_id -> Future

    def should_check(self, track_id: int, frame_idx: int) -> bool:
        if self.model is None:
            return False
        with self._lock:
            if track_id in self.active_violations:
                return False
            if track_id in self.pending_futures:
                return False
            if frame_idx % config.HELMET_CHECK_EVERY_N != 0:
                return False
            if self.attempts.get(track_id, 0) >= 5:
                return False
            return True

    def submit(
        self,
        track_id: int,
        vehicle_crop: np.ndarray,
        vehicle_class: str,
        plate: str,
        timestamp: str,
        vehicle_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> None:
        if self.model is None or vehicle_crop is None or vehicle_crop.size == 0:
            return

        with self._lock:
            self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
            future = self.executor.submit(
                self._run_detection,
                track_id,
                vehicle_crop,
                vehicle_class,
                plate,
                timestamp,
                vehicle_bbox
            )
            self.pending_futures[track_id] = future

    def _run_detection(
        self,
        track_id: int,
        vehicle_crop: np.ndarray,
        vehicle_class: str,
        plate: str,
        timestamp: str,
        vehicle_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[List[Dict]]:
        try:
            if self.model is None:
                return None

            results = self.model.predict(
                vehicle_crop,
                conf=config.HELMET_CONF_THRESHOLD,
                half=config.USE_HALF,
                device=self.device,
                verbose=False
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            detected_persons = []
            for i in range(len(xyxy)):
                cls_name = self.model.names[cls_ids[i]]
                x1, y1, x2, y2 = xyxy[i]
                x_center = (x1 + x2) / 2
                detected_persons.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "x_center": x_center,
                    "class_name": cls_name,
                    "conf": float(confs[i])
                })

            # Sort leftmost to rightmost by x_center
            detected_persons.sort(key=lambda p: p["x_center"])

            violations = []
            vx1, vy1, vx2, vy2 = vehicle_bbox if vehicle_bbox else (0, 0, 0, 0)

            # Rider (leftmost / first person)
            if len(detected_persons) >= 1:
                rider = detected_persons[0]
                mapped = config.HELMET_CLASS_MAP.get(rider["class_name"], "helmet")
                if mapped == "no_helmet":
                    rx1, ry1, rx2, ry2 = rider["bbox"]
                    violations.append({
                        "track_id": track_id,
                        "plate": plate,
                        "vehicle_class": vehicle_class,
                        "violation_type": "rider_no_helmet",
                        "timestamp": timestamp,
                        "frame_violation": True,
                        "person_bbox": (vx1 + rx1, vy1 + ry1, vx1 + rx2, vy1 + ry2)
                    })

            # Pillion (second person)
            if len(detected_persons) >= 2:
                pillion = detected_persons[1]
                mapped = config.HELMET_CLASS_MAP.get(pillion["class_name"], "helmet")
                if mapped == "no_helmet":
                    px1, py1, px2, py2 = pillion["bbox"]
                    violations.append({
                        "track_id": track_id,
                        "plate": plate,
                        "vehicle_class": vehicle_class,
                        "violation_type": "pillion_no_helmet",
                        "timestamp": timestamp,
                        "frame_violation": True,
                        "person_bbox": (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
                    })

            return violations

        except Exception as exc:
            print(f"[HelmetChecker] Error checking track {track_id}: {exc}")
            return None

    def drain_completed(self) -> List[Dict]:
        new_violations = []
        with self._lock:
            completed_ids = []
            for track_id, future in list(self.pending_futures.items()):
                if future.done():
                    try:
                        viols = future.result()
                        if viols:
                            # Save to active violations and all session logs
                            self.active_violations[track_id] = viols
                            for v in viols:
                                self.all_violations.append(v)
                                new_violations.append(v)
                    except Exception as exc:
                        print(f"[HelmetChecker] Future error for track {track_id}: {exc}")
                    completed_ids.append(track_id)

            for track_id in completed_ids:
                self.pending_futures.pop(track_id, None)

        return new_violations

    def get_active_violations(self) -> List[Dict]:
        with self._lock:
            flat_viols = []
            for viols in self.active_violations.values():
                flat_viols.extend(viols)
            return flat_viols

    def cleanup_stale(self, active_track_ids: Set[int]) -> None:
        with self._lock:
            for tid in list(self.active_violations.keys()):
                if tid not in active_track_ids:
                    self.active_violations.pop(tid, None)
            for tid in list(self.attempts.keys()):
                if tid not in active_track_ids:
                    self.attempts.pop(tid, None)
            for tid in list(self.pending_futures.keys()):
                if tid not in active_track_ids:
                    self.pending_futures.pop(tid, None)

    def shutdown(self) -> None:
        print("[HelmetChecker] Shutting down thread pool executor...")
        self.executor.shutdown(wait=False)
