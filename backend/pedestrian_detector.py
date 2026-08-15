"""Pedestrian Detector & Gender/Child Vision Engine.
Filters COCO person boxes using IoU overlap against vehicles, and classifies pedestrians asynchronously."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from ultralytics import YOLO

import config
from geometry_utils import calculate_iou, is_child_by_height


class RawBox:
    def __init__(self, cls: int, track_id: int, bbox: Tuple[int, int, int, int]):
        self.cls = cls
        self.track_id = track_id
        self.bbox = bbox


class PedestrianDetector:
    def __init__(self, device: str):
        self.device = device
        self._lock = threading.Lock()

        gender_path = Path(config.GENDER_MODEL)
        self.model = None
        self.deepface_available = False

        if not gender_path.exists():
            print(f"[PedestrianDetector] WARNING: Gender model not found at {gender_path}. Trying silent DeepFace fallback...")
            try:
                import importlib.util
                if importlib.util.find_spec("deepface") is None:
                    raise ImportError("deepface not installed")
                self.deepface_available = True
                print("[PedestrianDetector] DeepFace fallback is available and initialized silently.")
            except ImportError:
                self.deepface_available = False
                print("[PedestrianDetector] WARNING: models/gender_detector.pt not found and deepface is not installed. Pedestrian classification will be skipped.")
        else:
            try:
                self.model = YOLO(str(gender_path))
                self.model.to(device)
                print(f"[PedestrianDetector] Loaded gender model on {device}")
            except Exception as exc:
                print(f"[PedestrianDetector] ERROR: Failed to load gender model: {exc}")

        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # State tracking
        self.results: Dict[int, str] = {}                  # track_id -> "male_adult" | "female_adult" | "child" | "unknown"
        self.attempts: Dict[int, int] = {}                  # track_id -> attempts count
        self.pending_futures: Dict[int, Future[Optional[Tuple[int, str, str]]]] = {} # track_id -> Future
        
        # Frame contexts
        self.current_heights: Dict[int, int] = {}           # track_id -> height in px
        self.current_bboxes: Dict[int, Tuple[int, int, int, int]] = {} # track_id -> bbox
        self.avg_adult_height: float = 0.0

    def should_check(self, track_id: int, frame_idx: int) -> bool:
        if self.model is None and not self.deepface_available:
            return False
        with self._lock:
            if track_id in self.pending_futures:
                return False
            if frame_idx % config.GENDER_CHECK_EVERY_N != 0:
                return False
            if self.attempts.get(track_id, 0) >= 5:
                return False
            return True

    def filter_pedestrians(self, person_boxes: List[RawBox], vehicle_boxes: List[Tuple[int, int, int, int]]) -> List[RawBox]:
        pedestrians = []
        for p in person_boxes:
            box_a = p.bbox
            is_pedestrian = True
            for v_box in vehicle_boxes:
                if calculate_iou(box_a, v_box) >= config.PEDESTRIAN_VEHICLE_IOU:
                    is_pedestrian = False
                    break
            if is_pedestrian:
                pedestrians.append(p)

        # Update average adult height and current frame contexts
        adult_heights = []
        with self._lock:
            for p in pedestrians:
                h = p.bbox[3] - p.bbox[1]
                self.current_heights[p.track_id] = h
                self.current_bboxes[p.track_id] = p.bbox
                
                # Check if already classified as adult to calculate avg height
                cls = self.results.get(p.track_id)
                if cls in ("male_adult", "female_adult"):
                    adult_heights.append(h)

            if adult_heights:
                self.avg_adult_height = sum(adult_heights) / len(adult_heights)
            else:
                # Fallback to average height of all current pedestrians
                all_heights = [p.bbox[3] - p.bbox[1] for p in pedestrians]
                self.avg_adult_height = sum(all_heights) / len(all_heights) if all_heights else 0.0

        return pedestrians

    def submit(self, track_id: int, person_crop: np.ndarray, timestamp: str) -> None:
        if (self.model is None and not self.deepface_available) or person_crop is None or person_crop.size == 0:
            return

        with self._lock:
            self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
            future = self.executor.submit(self._run_classification, track_id, person_crop, timestamp)
            self.pending_futures[track_id] = future

    def _run_classification(self, track_id: int, person_crop: np.ndarray, timestamp: str) -> Optional[Tuple[int, str, str]]:
        try:
            gender = "unknown"
            
            if self.model is not None:
                results = self.model.predict(
                    person_crop,
                    conf=config.GENDER_CONF_THRESHOLD,
                    half=config.USE_HALF,
                    device=self.device,
                    verbose=False
                )
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    best_idx = int(results[0].boxes.conf.argmax().item())
                    cls_id = int(results[0].boxes.cls[best_idx].item())
                    cls_name = self.model.names[cls_id].lower()
                    
                    if "female" in cls_name:
                        gender = "female_adult"
                    elif "male" in cls_name:
                        gender = "male_adult"
                    elif "child" in cls_name or "kid" in cls_name:
                        gender = "child"

            elif self.deepface_available:
                try:
                    from deepface import DeepFace
                    # Runs deepface silently to retrieve gender/age
                    objs = DeepFace.analyze(
                        img_path=person_crop,
                        actions=['gender', 'age'],
                        enforce_detection=False,
                        silent=True
                    )
                    if objs:
                        obj = objs[0]
                        df_gender = obj.get("dominant_gender", "").lower()
                        df_age = obj.get("age", 30)
                        
                        if df_age < 12:
                            gender = "child"
                        elif "female" in df_gender:
                            gender = "female_adult"
                        elif "male" in df_gender:
                            gender = "male_adult"
                except Exception:
                    pass

            return track_id, gender, timestamp

        except Exception as exc:
            print(f"[PedestrianDetector] Error classifying track {track_id}: {exc}")
            return None

    def drain_completed(self) -> None:
        with self._lock:
            completed_ids = []
            for track_id, future in list(self.pending_futures.items()):
                if future.done():
                    try:
                        res = future.result()
                        if res is not None:
                            tid, gender, timestamp = res
                            
                            # Scene-relative child heuristic (documented limitation)
                            height = self.current_heights.get(tid, 0)
                            avg_adult = self.avg_adult_height
                            if is_child_by_height(height, avg_adult, config.CHILD_HEIGHT_RATIO):
                                gender = "child"
                                
                            self.results[tid] = gender
                    except Exception as exc:
                        print(f"[PedestrianDetector] Future error for track {track_id}: {exc}")
                    completed_ids.append(track_id)

            for track_id in completed_ids:
                self.pending_futures.pop(track_id, None)

    def get_current_pedestrians(self) -> Tuple[Dict, List[Dict]]:
        with self._lock:
            active_ids = list(self.current_bboxes.keys())
            details = []
            males = 0
            females = 0
            children = 0
            
            for tid in active_ids:
                gender = self.results.get(tid, "unknown")
                if gender == "male_adult":
                    males += 1
                elif gender == "female_adult":
                    females += 1
                elif gender == "child":
                    children += 1

                details.append({
                    "track_id": tid,
                    "gender": gender,
                    "timestamp": datetime.now().isoformat(),
                    "bbox": self.current_bboxes[tid]
                })

            total = len(active_ids)
            summary = {
                "total": total,
                "males": males,
                "females": females,
                "children": children
            }
            return summary, details

    def cleanup_stale(self, active_person_ids: Set[int]) -> None:
        with self._lock:
            for tid in list(self.results.keys()):
                if tid not in active_person_ids:
                    self.results.pop(tid, None)
            for tid in list(self.attempts.keys()):
                if tid not in active_person_ids:
                    self.attempts.pop(tid, None)
            for tid in list(self.pending_futures.keys()):
                if tid not in active_person_ids:
                    self.pending_futures.pop(tid, None)
            for tid in list(self.current_heights.keys()):
                if tid not in active_person_ids:
                    self.current_heights.pop(tid, None)
            for tid in list(self.current_bboxes.keys()):
                if tid not in active_person_ids:
                    self.current_bboxes.pop(tid, None)

    def shutdown(self) -> None:
        print("[PedestrianDetector] Shutting down thread pool executor...")
        self.executor.shutdown(wait=False)
