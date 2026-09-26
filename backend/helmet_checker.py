"""Helmet Checker — Handles legacy checkpoint aliasing and asynchronous helmet compliance classification."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import cv2
import numpy as np

# --- Fix legacy Ultralytics namespace mappings for older checkpoints ---
try:
    import ultralytics
    import ultralytics.nn
    import ultralytics.models
    # Alias missing legacy module paths
    sys.modules.setdefault("ultralytics.yolo", ultralytics)
    sys.modules.setdefault("ultralytics.yolo.nn", ultralytics.nn)
    sys.modules.setdefault("ultralytics.yolo.models", ultralytics.models)
except Exception:
    pass

from ultralytics import YOLO
import config


class HelmetChecker:
    def __init__(self, device: str):
        self.device = device
        self._lock = threading.Lock()
        self.model = None

        helmet_path = Path(config.HELMET_MODEL)
        if helmet_path.exists():
            try:
                self.model = YOLO(str(helmet_path))
                self.model.to(device)
                print(f"[HelmetChecker] Loaded helmet model on {device}")
            except Exception as exc:
                print(f"[HelmetChecker] WARNING: Failed to load helmet model: {exc}")
        else:
            print(f"[HelmetChecker] WARNING: Helmet model not found at {helmet_path}")

        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # State tracking
        self.all_two_wheeler_statuses: Dict[int, Dict[str, Any]] = {}
        self.active_violations: Dict[int, Dict[str, Any]] = {}
        self.attempts: Dict[int, int] = {}
        self.pending_futures: Dict[int, Future] = {}

    def should_check(self, track_id: int, frame_idx: int) -> bool:
        if self.model is None:
            return False
        with self._lock:
            if track_id in self.pending_futures:
                return False
            if frame_idx % config.HELMET_CHECK_EVERY_N != 0:
                return False
            if self.attempts.get(track_id, 0) >= 8:
                return False
            return True

    def submit(
        self,
        track_id: int,
        vehicle_crop: np.ndarray,
        vehicle_class: str,
        plate: str,
        timestamp: str,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        if self.model is None or vehicle_crop is None or vehicle_crop.size == 0:
            return

        with self._lock:
            self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
            future = self.executor.submit(
                self._run_inference, track_id, vehicle_crop.copy(), vehicle_class, plate, timestamp, bbox
            )
            self.pending_futures[track_id] = future

    def _run_inference(
        self,
        track_id: int,
        vehicle_crop: np.ndarray,
        vehicle_class: str,
        plate: str,
        timestamp: str,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[int, str, str, str]]:
        try:
            results = self.model.predict(
                vehicle_crop,
                conf=config.HELMET_CONF_THRESHOLD,
                device=self.device,
                verbose=False,
            )
            
            rider_status = "helmet"  # Default compliant unless violation detected
            
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                
                for cid in cls_ids:
                    cname = self.model.names[cid].lower()
                    mapped = config.HELMET_CLASS_MAP.get(cname, cname)
                    if "no_helmet" in mapped or "without_helmet" in mapped or "head" in mapped:
                        rider_status = "no_helmet"
                        break

            return track_id, rider_status, plate, timestamp

        except Exception as exc:
            # Fallback on inference error without crashing
            return track_id, "unknown", plate, timestamp

    def drain_completed(self) -> None:
        with self._lock:
            completed_ids = []
            for track_id, future in list(self.pending_futures.items()):
                if future.done():
                    try:
                        res = future.result()
                        if res is not None:
                            tid, status, plate, ts = res
                            if tid not in self.all_two_wheeler_statuses:
                                self.all_two_wheeler_statuses[tid] = {}
                            
                            self.all_two_wheeler_statuses[tid]["rider_helmet"] = status
                            self.all_two_wheeler_statuses[tid]["timestamp"] = ts
                            if plate != "UNKNOWN":
                                self.all_two_wheeler_statuses[tid]["plate"] = plate

                            if status == "no_helmet":
                                self.active_violations[tid] = {
                                    "track_id": tid,
                                    "type": "no_helmet",
                                    "timestamp": ts,
                                    "plate": plate,
                                }
                            elif status == "helmet" and tid in self.active_violations:
                                self.active_violations.pop(tid, None)

                    except Exception as exc:
                        print(f"[HelmetChecker] Future error for track {track_id}: {exc}")
                    completed_ids.append(track_id)

            for track_id in completed_ids:
                self.pending_futures.pop(track_id, None)

    def get_active_violations(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.active_violations.values())

    def cleanup_stale(self, active_track_ids: Set[int]) -> None:
        with self._lock:
            for tid in list(self.active_violations.keys()):
                if tid not in active_track_ids:
                    self.active_violations.pop(tid, None)
            for tid in list(self.pending_futures.keys()):
                if tid not in active_track_ids:
                    self.pending_futures.pop(tid, None)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)