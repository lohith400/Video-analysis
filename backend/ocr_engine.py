"""
OCR Engine — Production Implementation.
Uses Bilateral Filtering + CLAHE + Lanczos3 Resampling and Multi-Frame Trajectory Voting.
"""

from __future__ import annotations

import collections
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
import cv2
import easyocr
import numpy as np
from detector import PlateDetector

import config
from plate_utils import (
    STD_PLATE_REGEX,
    BH_PLATE_REGEX,
    PositionalRegexCorrector,
    levenshtein_distance,
    sort_character_boxes_multiline,
    clean_and_correct_indian_plate,
)


def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """
    Standardizes plate resolution and contrast.
    Bilateral Filter + CLAHE + Lanczos3 Interpolation.
    """
    if crop is None or crop.size == 0:
        return crop

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Bilateral filter preserves sharp character edges while removing noise
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)

    # CLAHE enhances local contrast under shadows and headlight glare
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    # Scale to standard height (64px) using Lanczos4 interpolation
    h, w = enhanced.shape
    target_h = 64
    target_w = int(w * (target_h / float(max(h, 1))))
    resized = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return resized


class OCREngine:
    def __init__(self, device: str, max_workers: int = config.OCR_MAX_WORKERS):
        self.device = device
        self._lock = threading.Lock()
        self.detector = PlateDetector(device)
        easyocr_dir = Path("models/easyocr")
        if not easyocr_dir.exists():
            alt_easy = Path("backend/models/easyocr")
            if alt_easy.exists():
                easyocr_dir = alt_easy
            else:
                alt_easy2 = Path("../models/easyocr")
                if alt_easy2.exists():
                    easyocr_dir = alt_easy2
        if easyocr_dir.exists():
            self.reader = easyocr.Reader(['en'], gpu=(device == "cuda"), model_storage_directory=str(easyocr_dir))
        else:
            self.reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # State tracking
        self.results: Dict[int, str] = {}
        self.plate_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        self.plate_confidences: Dict[int, float] = {}
        self.pending_futures: Dict[int, Future] = {}
        self.attempts: Dict[int, int] = {}
        
        # Trajectory observations: track_id -> list of observation dicts
        self.track_observations: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        print(f"[OCREngine] Initialized on {device} with {max_workers} thread pool workers.")

    def detect_plates_sync(
        self,
        frame: np.ndarray,
        vehicles: List,
    ) -> None:
        """Runs plate localization synchronously on current frame and registers observations."""
        if not vehicles or frame is None or frame.size == 0:
            return

        try:
            results = self.detector.model.predict(
                frame,
                conf=config.PLATE_CONF_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                imgsz=1280,
                half=config.PLATE_USE_HALF,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            print(f"[OCREngine] Plate detection error: {exc}")
            return

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):
            px1, py1, px2, py2 = map(int, xyxy[i])
            det_conf = float(confs[i])

            best_vehicle = None
            best_overlap = 0.0
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v.bbox
                ix1, iy1 = max(px1, vx1), max(py1, vy1)
                ix2, iy2 = min(px2, vx2), min(py2, vy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                plate_area = max(1, (px2 - px1) * (py2 - py1))
                overlap = inter / plate_area
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_vehicle = v

            if best_vehicle is None or best_overlap < 0.40:
                continue

            track_id = best_vehicle.track_id
            
            with self._lock:
                self.plate_boxes[track_id] = (px1, py1, px2, py2)

            if not self.needs_ocr(track_id):
                continue

            # Crop with safe margin
            frame_h, frame_w = frame.shape[:2]
            box_w, box_h = px2 - px1, py2 - py1
            pad_x = max(6, int(box_w * 0.15))
            pad_y = max(6, int(box_h * 0.25))
            crop_x1 = max(0, px1 - pad_x)
            crop_y1 = max(0, py1 - pad_y)
            crop_x2 = min(frame_w, px2 + pad_x)
            crop_y2 = min(frame_h, py2 + pad_y)
            plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if plate_crop.size == 0:
                continue

            # OCR inference with spatial sorting
            raw_text, ocr_conf = self._read_plate_text(plate_crop)

            if raw_text:
                with self._lock:
                    self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
                    bbox_area = float((px2 - px1) * (py2 - py1))
                    
                    self.track_observations[track_id].append({
                        'raw_text': raw_text,
                        'det_conf': det_conf,
                        'ocr_conf': ocr_conf,
                        'bbox_area': bbox_area
                    })

                    # Resolve candidate via Weighted Temporal Voting
                    best_plate, final_conf = self._aggregate_trajectory(track_id)
                    if best_plate:
                        self.results[track_id] = best_plate
                        self.plate_confidences[track_id] = final_conf

    def _read_plate_text(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Runs EasyOCR, applies multi-line spatial sorting, and computes confidence score."""
        try:
            processed = preprocess_plate_crop(plate_crop)
            ocr_results = self.reader.readtext(
                processed,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,
                paragraph=False
            )
            if not ocr_results:
                return None, 0.0

            # Sort character fragments into proper reading order
            sorted_results = sort_character_boxes_multiline(ocr_results)
            
            raw_text = "".join(r[1] for r in sorted_results)
            raw_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
            
            avg_conf = float(np.mean([r[2] for r in sorted_results])) if sorted_results else 0.0
            return (raw_text, avg_conf) if len(raw_text) >= 4 else (None, 0.0)
        except Exception:
            return None, 0.0

    def _aggregate_trajectory(self, track_id: int) -> Tuple[Optional[str], float]:
        """Weighted Levenshtein distance temporal voting."""
        observations = self.track_observations.get(track_id, [])
        if not observations:
            return None, 0.0

        clusters = defaultdict(list)
        for obs in observations:
            corrected = PositionalRegexCorrector.correct_string(obs['raw_text'])
            if not corrected:
                continue

            matched_cluster = None
            for centroid in clusters.keys():
                if levenshtein_distance(corrected, centroid) <= 2:
                    matched_cluster = centroid
                    break

            if matched_cluster:
                clusters[matched_cluster].append((corrected, obs))
            else:
                clusters[corrected].append((corrected, obs))

        best_plate = None
        max_score = -1.0
        best_conf = 0.0

        for centroid, item_list in clusters.items():
            cluster_score = 0.0
            candidate_counts = defaultdict(float)
            candidate_confs = defaultdict(list)

            for string_val, obs in item_list:
                # Formula: Weight = DetConf * OCRConf * BBoxArea
                weight = obs['det_conf'] * obs['ocr_conf'] * (obs['bbox_area'] ** 0.5)
                cluster_score += weight
                candidate_counts[string_val] += weight
                candidate_confs[string_val].append(obs['ocr_conf'])

            if cluster_score > max_score and len(item_list) >= 2:
                max_score = cluster_score
                best_plate = max(candidate_counts, key=candidate_counts.get)
                confs = candidate_confs[best_plate]
                best_conf = float(np.mean(confs)) if confs else 0.85

        return best_plate, round(best_conf, 2)

    def needs_ocr(self, track_id: int) -> bool:
        with self._lock:
            existing = self.results.get(track_id)
            if existing and (STD_PLATE_REGEX.match(existing) or BH_PLATE_REGEX.match(existing)):
                return False
            if self.attempts.get(track_id, 0) >= 15:
                return False
            return True

    def get_plate(self, track_id: int) -> Optional[str]:
        with self._lock:
            return self.results.get(track_id)

    def get_all_plates(self) -> Dict[int, str]:
        with self._lock:
            return dict(self.results)

    def get_plate_box(self, track_id: int) -> Optional[Tuple[int, int, int, int]]:
        with self._lock:
            return self.plate_boxes.get(track_id)

    def get_all_plate_boxes(self) -> Dict[int, Tuple[int, int, int, int]]:
        with self._lock:
            return dict(self.plate_boxes)

    @property
    def total_plates_detected(self) -> int:
        with self._lock:
            return len(self.results)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)