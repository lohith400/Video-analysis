"""
Pedestrian Detector & Demographic Vision Engine (IRIS v2.1).
Features:
- Dedicated COCO-pretrained person detection & ByteTrack tracking (decoupled from vehicle model).
- Upgraded rider exclusion: intersection / person_area >= 0.50 + upward-expanded two-wheeler box containment.
- Modular DemographicClassifier interface: classify(crop) -> (label, confidence).
- Strict quality gates: min height (80px), min width (30px), Laplacian sharpness, edge rejection.
- Temporal confidence-weighted voting per track (15-frame history, >=60% consensus, >=3 votes) to prevent flip-flopping.
- Model priority over weak child-height heuristic (heuristic disabled by default).
- Rolling median count smoothing to eliminate instantaneous CSV snapshot spikes.
- Loud error reporting at startup if model weights are missing.
"""

from __future__ import annotations

import sys
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config
from geometry_utils import (
    calculate_iou,
    is_rider_of_vehicle,
    resolve_demographic_label,
    TrackVoteAccumulator,
    RollingMedianSmoother,
)


class RawBox:
    def __init__(self, cls: int, track_id: int, bbox: Tuple[int, int, int, int]):
        self.cls = cls
        self.track_id = track_id
        self.bbox = bbox


class DemographicClassifier:
    """
    Standard demographic classification interface.
    classify(crop) -> (label, confidence)
    Labels: 'male_adult', 'female_adult', 'child', 'unknown'
    """

    def __init__(self, device: str = "cpu", model_path: Optional[str] = None):
        self.device = device
        self.model = None

        target_path = Path(model_path or config.GENDER_MODEL)
        resolved_path = None
        for candidate in [
            target_path,
            Path("backend") / target_path,
            Path("..") / target_path,
            Path("models/gender_detector.pt"),
            Path("backend/models/gender_detector.pt"),
        ]:
            if candidate.exists():
                resolved_path = candidate
                break

        if resolved_path is None or not resolved_path.exists():
            print(
                f"[DemographicClassifier] ERROR: Demographic model weights not found at {target_path}. "
                f"Demographic classification disabled — active fallback: abstain as 'unknown'.",
                file=sys.stderr,
            )
        else:
            try:
                self.model = YOLO(str(resolved_path))
                self.model.to(device)
                print(f"[DemographicClassifier] Loaded demographic model from {resolved_path} on {device}")
            except Exception as exc:
                print(
                    f"[DemographicClassifier] ERROR: Failed to load demographic model at {resolved_path}: {exc}. "
                    f"Active fallback: abstain as 'unknown'.",
                    file=sys.stderr,
                )
                self.model = None

    def check_quality(
        self,
        crop: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Quality gates before classification: min size, blur check, and edge rejection."""
        if crop is None or crop.size == 0:
            return False

        h, w = crop.shape[:2]
        if h < config.MIN_CROP_HEIGHT or w < config.MIN_CROP_WIDTH:
            return False

        # Edge crop rejection: person cut off by camera frame boundary
        if bbox is not None and frame_shape is not None:
            fh, fw = frame_shape[:2]
            bx1, by1, bx2, by2 = bbox
            if bx1 <= 2 or by1 <= 2 or bx2 >= fw - 2 or by2 >= fh - 2:
                return False

        # Sharpness check via Laplacian variance
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < config.MIN_CROP_SHARPNESS:
                return False
        except Exception:
            return False

        return True

    def classify(
        self,
        crop: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        height: float = 0.0,
        avg_adult_height: float = 0.0,
    ) -> Tuple[str, float]:
        """Classify a pedestrian crop into (label, confidence)."""
        if not self.check_quality(crop, bbox=bbox, frame_shape=frame_shape):
            return "unknown", 0.0

        if self.model is None:
            return "unknown", 0.0

        try:
            results = self.model.predict(crop, device=self.device, verbose=False)
            if not results:
                return "unknown", 0.0

            res = results[0]
            raw_label = ""
            conf = 0.0

            if hasattr(res, "probs") and res.probs is not None:
                probs = res.probs
                top1_idx = int(probs.top1)
                raw_label = self.model.names[top1_idx].lower()
                conf = float(probs.top1conf)
            elif res.boxes is not None and len(res.boxes) > 0:
                best_idx = int(res.boxes.conf.argmax().item())
                cls_id = int(res.boxes.cls[best_idx].item())
                raw_label = self.model.names[cls_id].lower()
                conf = float(res.boxes.conf[best_idx].item())
            else:
                return "unknown", 0.0

            label, resolved_conf = resolve_demographic_label(
                model_label=raw_label,
                model_conf=conf,
                height=height,
                avg_adult_height=avg_adult_height,
                use_height_heuristic=config.USE_HEIGHT_CHILD_HEURISTIC,
                child_height_ratio=config.CHILD_HEIGHT_RATIO,
                conf_threshold=config.GENDER_CONF_THRESHOLD,
            )
            return label, resolved_conf

        except Exception as exc:
            print(f"[DemographicClassifier] Inference error: {exc}", file=sys.stderr)
            return "unknown", 0.0


class PedestrianDetector:
    """
    Manages dedicated person detection, persistence tracking, rider exclusion,
    temporal demographic voting, and count smoothing.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._lock = threading.Lock()

        # 1. Demographic Classifier
        self.classifier = DemographicClassifier(device=device)

        # 2. Dedicated Person Tracker (COCO YOLO, class 0)
        self.person_model = None
        detector_path = Path(config.PEDESTRIAN_MODEL)
        resolved_detector = None
        for candidate in [
            detector_path,
            Path("backend") / detector_path,
            Path("..") / detector_path,
            Path("models/yolov8n.pt"),
            Path("backend/models/yolov8n.pt"),
            Path("yolov8n.pt"),
        ]:
            if candidate.exists():
                resolved_detector = candidate
                break

        if resolved_detector is None:
            print(
                f"[PedestrianDetector] ERROR: Dedicated pedestrian model not found at {detector_path}. "
                f"Falling back to Ultralytics yolov8n.pt auto-download.",
                file=sys.stderr,
            )
            try:
                self.person_model = YOLO("yolov8n.pt")
                self.person_model.to(device)
            except Exception as e:
                print(f"[PedestrianDetector] FATAL: Cannot initialize pedestrian detector: {e}", file=sys.stderr)
        else:
            try:
                self.person_model = YOLO(str(resolved_detector))
                self.person_model.to(device)
                print(f"[PedestrianDetector] Dedicated pedestrian tracker loaded from {resolved_detector} on {device}")
            except Exception as e:
                print(f"[PedestrianDetector] ERROR: Failed loading {resolved_detector}: {e}", file=sys.stderr)

        self.executor = ThreadPoolExecutor(max_workers=2)

        # 3. Temporal Voting & Smoothing Helpers
        self.vote_accumulator = TrackVoteAccumulator(
            window_size=config.VOTE_WINDOW,
            min_votes=config.MIN_ACCEPTED_VOTES,
            weight_threshold=config.VOTE_WEIGHT_THRESHOLD,
        )
        self.count_smoother = RollingMedianSmoother(
            window_duration_sec=config.PEDESTRIAN_SMOOTHING_WINDOW_SEC
        )

        # 4. State Tracking
        self.track_ages: Dict[int, int] = {}
        self.results: Dict[int, str] = {}
        self.attempts: Dict[int, int] = {}
        self.pending_futures: Dict[int, Future[Optional[Tuple[int, str, float, str]]]] = {}

        # Frame Contexts
        self.current_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        self.current_heights: Dict[int, int] = {}
        self.avg_adult_height: float = 0.0

        # Unique Stable Pedestrians (Session totals)
        self.all_stable_pedestrians: Dict[int, str] = {}

    def detect_and_track(
        self,
        frame: np.ndarray,
        vehicle_boxes: List[Tuple[int, int, int, int]],
        vehicle_classes: Optional[List[str]] = None,
    ) -> List[RawBox]:
        """
        Runs dedicated person tracking on the frame and filters out riders.
        """
        if self.person_model is None or frame is None:
            return []

        h_frame, w_frame = frame.shape[:2]
        try:
            results = self.person_model.track(
                frame,
                persist=True,
                classes=[0],
                tracker=config.TRACKER_CONFIG,
                conf=config.PEDESTRIAN_CONF_THRESHOLD,
                verbose=False,
            )
        except Exception as exc:
            print(f"[PedestrianDetector] Tracking error: {exc}", file=sys.stderr)
            return []

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            with self._lock:
                self.current_bboxes.clear()
                self.current_heights.clear()
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        raw_candidates: List[RawBox] = []
        for i in range(len(xyxy)):
            box = (int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]))
            raw_candidates.append(RawBox(cls=0, track_id=track_ids[i], bbox=box))

        return self.filter_pedestrians(raw_candidates, vehicle_boxes, vehicle_classes, frame_shape=(h_frame, w_frame))

    def filter_pedestrians(
        self,
        person_boxes: List[RawBox],
        vehicle_boxes: List[Tuple[int, int, int, int]],
        vehicle_classes: Optional[List[str]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[RawBox]:
        """
        Applies rider exclusion geometry and track stability gates.
        """
        v_classes = vehicle_classes or [""] * len(vehicle_boxes)
        valid_pedestrians: List[RawBox] = []

        with self._lock:
            active_frame_bboxes = {}
            active_frame_heights = {}
            adult_heights = []

            for p in person_boxes:
                p_box = p.bbox
                ph = p_box[3] - p_box[1]
                tid = p.track_id

                # Box height filter (reject tiny artifacts)
                if ph < config.MIN_PEDESTRIAN_HEIGHT:
                    continue

                # Rider exclusion check against all vehicles
                is_rider = False
                for v_box, v_cls in zip(vehicle_boxes, v_classes):
                    if is_rider_of_vehicle(
                        person_box=p_box,
                        vehicle_box=v_box,
                        vehicle_class=v_cls,
                        overlap_thresh=config.PEDESTRIAN_RIDER_OVERLAP,
                        upward_expansion_ratio=config.RIDER_BOX_UPWARD_EXPANSION,
                    ):
                        is_rider = True
                        break

                if is_rider:
                    continue

                # Increment track age
                self.track_ages[tid] = self.track_ages.get(tid, 0) + 1
                active_frame_bboxes[tid] = p_box
                active_frame_heights[tid] = ph
                valid_pedestrians.append(p)

                # Register stable track in session registry once track age threshold reached
                if self.track_ages[tid] >= config.MIN_PEDESTRIAN_TRACK_AGE:
                    current_label = self.results.get(tid, "unknown")
                    self.all_stable_pedestrians[tid] = current_label

                # Accumulate adult heights for weak child reference
                if self.results.get(tid) in ("male_adult", "female_adult"):
                    adult_heights.append(ph)

            self.current_bboxes = active_frame_bboxes
            self.current_heights = active_frame_heights

            if adult_heights:
                self.avg_adult_height = sum(adult_heights) / len(adult_heights)
            elif active_frame_heights:
                self.avg_adult_height = sum(active_frame_heights.values()) / len(active_frame_heights)

        return valid_pedestrians

    def should_check(self, track_id: int, frame_idx: int) -> bool:
        if self.classifier.model is None:
            return False
        with self._lock:
            if track_id in self.pending_futures:
                return False
            if frame_idx % config.GENDER_CHECK_EVERY_N != 0:
                return False
            if self.attempts.get(track_id, 0) >= 10:
                return False
            return True

    def submit(
        self,
        track_id: int,
        person_crop: np.ndarray,
        timestamp: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        if self.classifier.model is None or person_crop is None or person_crop.size == 0:
            return

        with self._lock:
            self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
            h = self.current_heights.get(track_id, 0)
            avg_h = self.avg_adult_height
            box = bbox or self.current_bboxes.get(track_id)

            future = self.executor.submit(
                self._run_classification,
                track_id,
                person_crop,
                timestamp,
                box,
                frame_shape,
                h,
                avg_h,
            )
            self.pending_futures[track_id] = future

    def _run_classification(
        self,
        track_id: int,
        person_crop: np.ndarray,
        timestamp: str,
        bbox: Optional[Tuple[int, int, int, int]],
        frame_shape: Optional[Tuple[int, int]],
        height: float,
        avg_adult_height: float,
    ) -> Optional[Tuple[int, str, float, str]]:
        try:
            label, conf = self.classifier.classify(
                crop=person_crop,
                bbox=bbox,
                frame_shape=frame_shape,
                height=height,
                avg_adult_height=avg_adult_height,
            )
            return track_id, label, conf, timestamp
        except Exception as exc:
            print(f"[PedestrianDetector] Error classifying track {track_id}: {exc}", file=sys.stderr)
            return None

    def drain_completed(self) -> None:
        with self._lock:
            completed_ids = []
            for track_id, future in list(self.pending_futures.items()):
                if future.done():
                    try:
                        res = future.result()
                        if res is not None:
                            tid, label, conf, timestamp = res
                            if label != "unknown" and conf >= config.GENDER_CONF_THRESHOLD:
                                self.vote_accumulator.add_vote(tid, label, conf)
                                voted_label, _ = self.vote_accumulator.get_voted_label(tid)
                                self.results[tid] = voted_label
                                if tid in self.all_stable_pedestrians:
                                    self.all_stable_pedestrians[tid] = voted_label
                    except Exception as exc:
                        print(f"[PedestrianDetector] Future error for track {track_id}: {exc}", file=sys.stderr)
                    completed_ids.append(track_id)

            for tid in completed_ids:
                self.pending_futures.pop(tid, None)

    def get_current_pedestrians(self) -> Tuple[Dict[str, int], List[Dict]]:
        """
        Returns instantaneous smoothed pedestrian counts and per-person details.
        Counts are smoothed via rolling median to prevent snapshot occlusion flicker.
        """
        with self._lock:
            active_ids = list(self.current_bboxes.keys())
            details = []
            raw_males = 0
            raw_females = 0
            raw_children = 0
            raw_unknown = 0

            for tid in active_ids:
                gender = self.results.get(tid, "unknown")
                if gender == "male_adult":
                    raw_males += 1
                elif gender == "female_adult":
                    raw_females += 1
                elif gender == "child":
                    raw_children += 1
                else:
                    raw_unknown += 1

                details.append({
                    "track_id": tid,
                    "gender": gender,
                    "timestamp": datetime.now().isoformat(),
                    "bbox": self.current_bboxes[tid],
                })

            raw_total = len(active_ids)
            now_sec = time.time()
            smoothed_total = self.count_smoother.update(now_sec, raw_total)

            summary = {
                "total": smoothed_total,
                "males": raw_males,
                "females": raw_females,
                "children": raw_children,
                "unknown": raw_unknown,
            }
            return summary, details

    def get_session_summary(self) -> Dict[str, int]:
        """
        Returns cumulative session unique stable pedestrian totals.
        """
        with self._lock:
            males = 0
            females = 0
            children = 0
            unknown = 0

            for tid, gender in self.all_stable_pedestrians.items():
                if gender == "male_adult":
                    males += 1
                elif gender == "female_adult":
                    females += 1
                elif gender == "child":
                    children += 1
                else:
                    unknown += 1

            total = len(self.all_stable_pedestrians)
            return {
                "total": total,
                "males": males,
                "females": females,
                "children": children,
                "unknown": unknown,
            }

    def cleanup_stale(self, active_person_ids: Set[int]) -> None:
        with self._lock:
            # Retain voting history and results for a grace period, clean up transient states
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