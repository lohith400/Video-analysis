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
        if not helmet_path.exists():
            alt = Path("backend") / config.HELMET_MODEL
            if alt.exists():
                helmet_path = alt
            else:
                alt2 = Path("..") / config.HELMET_MODEL
                if alt2.exists():
                    helmet_path = alt2

        if helmet_path.exists():
            try:
                self.model = YOLO(str(helmet_path))
                self.model.to(device)
                print(f"[HelmetChecker] Loaded helmet model from {helmet_path} on {device}")
            except Exception as exc:
                print(f"[HelmetChecker] WARNING: Failed to load helmet model: {exc}")
        else:
            print(f"[HelmetChecker] WARNING: Helmet model not found at {helmet_path}")

        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # State tracking
        self.all_two_wheeler_statuses: Dict[int, Dict[str, Any]] = {}
        self.active_violations: Dict[int, Dict[str, Any]] = {}
        self.all_time_violations: Dict[int, List[Dict[str, Any]]] = {}
        self.violation_history: Set[int] = set()
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
            # Check up to 20 times across the trajectory for high confidence
            if self.attempts.get(track_id, 0) >= 20:
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
    ) -> Optional[Dict[str, Any]]:
        try:
            results = self.model.predict(
                vehicle_crop,
                conf=config.HELMET_CONF_THRESHOLD,
                device=self.device,
                verbose=False,
            )
            
            heads = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                vx1, vy1 = bbox[0], bbox[1]

                for i in range(len(xyxy)):
                    hx1, hy1, hx2, hy2 = map(int, xyxy[i])
                    cid = cls_ids[i]
                    cname = self.model.names[cid].lower()
                    mapped = config.HELMET_CLASS_MAP.get(cname, cname)
                    is_no_helmet = ("no_helmet" in mapped or "without_helmet" in mapped or "head" in mapped)
                    
                    # Convert to absolute frame coordinates
                    abs_box = (vx1 + hx1, vy1 + hy1, vx1 + hx2, vy1 + hy2)
                    heads.append({
                        "crop_bbox": (hx1, hy1, hx2, hy2),
                        "abs_bbox": abs_box,
                        "status": "no_helmet" if is_no_helmet else "helmet",
                        "conf": float(confs[i]),
                        "center_x": (hx1 + hx2) / 2.0,
                        "center_y": (hy1 + hy2) / 2.0,
                    })

            # Sort heads along X-axis
            heads.sort(key=lambda h: h["center_x"])

            rider_status = "unknown"
            pillion_status = "none"
            violations = []

            if len(heads) == 1:
                rider_status = heads[0]["status"]
                pillion_status = "none"
                if rider_status == "no_helmet":
                    violations.append({
                        "track_id": track_id,
                        "type": "no_helmet_rider",
                        "role": "Rider",
                        "timestamp": timestamp,
                        "plate": plate,
                        "person_bbox": heads[0]["abs_bbox"],
                    })
            elif len(heads) >= 2:
                # First head is rider, second head is pillion
                rider_status = heads[0]["status"]
                pillion_status = heads[1]["status"]
                if rider_status == "no_helmet":
                    violations.append({
                        "track_id": track_id,
                        "type": "no_helmet_rider",
                        "role": "Rider",
                        "timestamp": timestamp,
                        "plate": plate,
                        "person_bbox": heads[0]["abs_bbox"],
                    })
                if pillion_status == "no_helmet":
                    violations.append({
                        "track_id": track_id,
                        "type": "no_helmet_pillion",
                        "role": "Pillion",
                        "timestamp": timestamp,
                        "plate": plate,
                        "person_bbox": heads[1]["abs_bbox"],
                    })

            return {
                "track_id": track_id,
                "rider_status": rider_status,
                "pillion_status": pillion_status,
                "plate": plate,
                "timestamp": timestamp,
                "violations": violations,
            }

        except Exception as exc:
            return {
                "track_id": track_id,
                "rider_status": "unknown",
                "pillion_status": "none",
                "plate": plate,
                "timestamp": timestamp,
                "violations": [],
            }

    def drain_completed(self) -> None:
        with self._lock:
            completed_ids = []
            for track_id, future in list(self.pending_futures.items()):
                if future.done():
                    try:
                        res = future.result()
                        if res is not None:
                            tid = res["track_id"]
                            r_status = res["rider_status"]
                            p_status = res["pillion_status"]
                            plate = res["plate"]
                            ts = res["timestamp"]
                            viols = res["violations"]

                            if tid not in self.all_two_wheeler_statuses:
                                self.all_two_wheeler_statuses[tid] = {
                                    "track_id": tid,
                                    "plate": plate if (plate and plate != "UNKNOWN") else "not detected",
                                    "vehicle_class": "Bike/Motorcycle",
                                    "rider_helmet": r_status,
                                    "pillion_helmet": p_status,
                                    "timestamp": ts,
                                    "has_violation": bool(viols),
                                    "violations": list(viols),
                                }
                            else:
                                cur = self.all_two_wheeler_statuses[tid]
                                if plate and plate != "UNKNOWN" and cur.get("plate") in ("not detected", "UNKNOWN"):
                                    cur["plate"] = plate

                                # Persist definitive rider status
                                if r_status == "no_helmet":
                                    cur["rider_helmet"] = "no_helmet"
                                    cur["has_violation"] = True
                                elif r_status == "helmet" and cur.get("rider_helmet") != "no_helmet":
                                    cur["rider_helmet"] = "helmet"

                                # Persist definitive pillion status
                                if p_status == "no_helmet":
                                    cur["pillion_helmet"] = "no_helmet"
                                    cur["has_violation"] = True
                                elif p_status == "helmet" and cur.get("pillion_helmet") != "no_helmet":
                                    cur["pillion_helmet"] = "helmet"
                                elif p_status != "none" and cur.get("pillion_helmet") == "none":
                                    cur["pillion_helmet"] = p_status

                            if viols:
                                self.active_violations[tid] = viols
                                self.all_time_violations[tid] = viols
                                self.violation_history.add(tid)
                                self.all_two_wheeler_statuses[tid]["has_violation"] = True
                                self.all_two_wheeler_statuses[tid]["violations"] = viols
                            elif (r_status == "helmet" and p_status in ("helmet", "none")) and tid in self.active_violations:
                                self.active_violations.pop(tid, None)

                    except Exception as exc:
                        print(f"[HelmetChecker] Future error for track {track_id}: {exc}")
                    completed_ids.append(track_id)

            for track_id in completed_ids:
                self.pending_futures.pop(track_id, None)

    def get_all_two_wheelers(self) -> Dict[int, Dict[str, Any]]:
        """Returns a snapshot of all two-wheeler states recorded during the session."""
        with self._lock:
            return {tid: dict(info) for tid, info in self.all_two_wheeler_statuses.items()}

    def get_active_violations(self) -> List[Dict[str, Any]]:
        with self._lock:
            all_v = []
            for item in self.active_violations.values():
                if isinstance(item, list):
                    all_v.extend(item)
                elif isinstance(item, dict):
                    all_v.append(item)
            return all_v

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