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
    extract_valid_plate,
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


def _upscale_min_height(img: np.ndarray, min_h: int) -> np.ndarray:
    """Cubic upscale so tiny two-wheeler plates reach a height EasyOCR can read."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or h >= min_h:
        return img
    f = min_h / float(h)
    return cv2.resize(img, (max(1, int(w * f)), min_h), interpolation=cv2.INTER_CUBIC)


def _plain_variant(crop: np.ndarray) -> np.ndarray:
    """Gentle variant (no CLAHE / bilateral): CLAHE amplifies noise on tiny low-res plates."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    gray = _upscale_min_height(gray, getattr(config, "PLATE_OCR_MIN_HEIGHT", 96))
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


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
        # Tracks whose published plate rests on a single observation (keep reading to confirm)
        self.provisional: set = set()
        self._frame_counter = 0
        
        print(f"[OCREngine] Initialized on {device} with {max_workers} thread pool workers.")

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _plausible_plate_box(x1: int, y1: int, x2: int, y2: int) -> bool:
        w, h = x2 - x1, y2 - y1
        if w <= 2 or h <= 2:
            return False
        ar = w / float(h)
        return getattr(config, "PLATE_MIN_ASPECT", 0.8) <= ar <= getattr(config, "PLATE_MAX_ASPECT", 6.5)

    def detect_plates_sync(
        self,
        frame: np.ndarray,
        vehicles: List,
    ) -> None:
        """
        Pass 1: full-frame plate detection (unchanged behaviour, works for cars/trucks/buses).
        Pass 2: vehicle-crop detection for two-wheelers / autos that pass 1 missed.
        All boxes are stored in absolute frame coordinates.
        """
        if not vehicles or frame is None or frame.size == 0:
            return
        self._frame_counter += 1
        matched_tracks: set = set()

        # ---------------- Pass 1: full frame ----------------
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
            results = None

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                px1, py1, px2, py2 = map(int, xyxy[i])
                det_conf = float(confs[i])
                if not self._plausible_plate_box(px1, py1, px2, py2):
                    continue

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

                matched_tracks.add(best_vehicle.track_id)
                self._handle_plate(frame, best_vehicle.track_id, (px1, py1, px2, py2), det_conf)

        # ---------------- Pass 2: vehicle crops ----------------
        if getattr(config, "PLATE_CROP_PASS_ENABLED", False):
            try:
                self._crop_pass(frame, vehicles, matched_tracks)
            except Exception as exc:  # never let the second pass break the pipeline
                print(f"[OCREngine] Crop-pass error: {exc}")

    def _crop_pass(self, frame: np.ndarray, vehicles: List, matched_tracks: set) -> None:
        crop_classes = set(getattr(config, "PLATE_CROP_PASS_CLASSES", []))
        every_n = max(1, int(getattr(config, "PLATE_CROP_EVERY_N_FRAMES", 1)))
        max_per_frame = int(getattr(config, "PLATE_CROP_MAX_PER_FRAME", 6))
        pad = float(getattr(config, "PLATE_CROP_PAD", 0.08))
        min_long = int(getattr(config, "PLATE_CROP_MIN_LONG_SIDE", 640))
        max_imgsz = int(getattr(config, "PLATE_CROP_MAX_IMGSZ", 960))
        crop_conf = float(getattr(config, "PLATE_CROP_CONF", 0.15))
        fh, fw = frame.shape[:2]

        candidates = [
            v for v in vehicles
            if v.track_id not in matched_tracks
            and str(v.vehicle_class).lower() in crop_classes
            and self.needs_ocr(v.track_id)
            and (self._frame_counter + v.track_id) % every_n == 0
        ]
        # biggest (closest) vehicles first: their plates are the most legible
        candidates.sort(key=lambda v: (v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1]), reverse=True)

        for v in candidates[:max_per_frame]:
            vx1, vy1, vx2, vy2 = v.bbox
            bw, bh = vx2 - vx1, vy2 - vy1
            if bw <= 8 or bh <= 8:
                continue
            cx1 = max(0, int(vx1 - bw * pad))
            cy1 = max(0, int(vy1 - bh * pad))
            cx2 = min(fw, int(vx2 + bw * pad))
            cy2 = min(fh, int(vy2 + bh * pad))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            scale = 1.0
            long_side = max(ch, cw)
            if long_side < min_long:
                scale = min_long / float(long_side)
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
            imgsz = int(min(max_imgsz, ((max(crop.shape[:2]) + 31) // 32) * 32))

            try:
                res = self.detector.model.predict(
                    crop,
                    conf=crop_conf,
                    iou=config.IOU_THRESHOLD,
                    imgsz=imgsz,
                    half=config.PLATE_USE_HALF,
                    device=self.device,
                    verbose=False,
                )
            except Exception as exc:
                print(f"[OCREngine] Crop plate detection error: {exc}")
                continue
            if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                continue

            rb = res[0].boxes
            rxyxy = rb.xyxy.cpu().numpy()
            rconf = rb.conf.cpu().numpy()
            best = None
            for j in range(len(rxyxy)):
                # map back to ABSOLUTE frame coordinates
                bx1 = int(rxyxy[j][0] / scale) + cx1
                by1 = int(rxyxy[j][1] / scale) + cy1
                bx2 = int(rxyxy[j][2] / scale) + cx1
                by2 = int(rxyxy[j][3] / scale) + cy1
                if not self._plausible_plate_box(bx1, by1, bx2, by2):
                    continue
                # plate centre must sit inside THIS vehicle (not a neighbour caught by the padding)
                pcx, pcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if not (vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2):
                    continue
                if best is None or float(rconf[j]) > best[1]:
                    best = ((bx1, by1, bx2, by2), float(rconf[j]))
            if best is not None:
                self._handle_plate(frame, v.track_id, best[0], best[1])

    def _handle_plate(self, frame: np.ndarray, track_id: int, box: Tuple[int, int, int, int], det_conf: float) -> None:
        """Store the plate box, run OCR on the native-resolution crop and update the vote."""
        px1, py1, px2, py2 = box
        with self._lock:
            self.plate_boxes[track_id] = (px1, py1, px2, py2)

        if not self.needs_ocr(track_id):
            return

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
            return

        raw_text, ocr_conf = self._read_plate_text(plate_crop)
        if not raw_text:
            return

        with self._lock:
            self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
            self.track_observations[track_id].append({
                'raw_text': raw_text,
                'det_conf': det_conf,
                'ocr_conf': ocr_conf,
                'bbox_area': float((px2 - px1) * (py2 - py1)),
            })
            best_plate, final_conf = self._aggregate_trajectory(track_id)
            if best_plate:
                self.results[track_id] = best_plate
                self.plate_confidences[track_id] = final_conf

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    def _ocr_image(self, img: np.ndarray) -> Tuple[str, float]:
        ocr_results = self.reader.readtext(
            img,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=1,
            paragraph=False,
        )
        if not ocr_results:
            return "", 0.0
        sorted_results = sort_character_boxes_multiline(ocr_results)
        text = re.sub(r"[^A-Z0-9]", "", "".join(r[1] for r in sorted_results).upper())
        conf = float(np.mean([r[2] for r in sorted_results])) if sorted_results else 0.0
        return text, conf

    def _read_two_line(self, variant: np.ndarray) -> Tuple[str, float]:
        """Two-wheeler plates are usually 2 lines (state+RTO / series+number). Read each half separately."""
        h = variant.shape[0]
        overlap = max(1, int(h * 0.04))
        top = variant[: h // 2 + overlap, :]
        bot = variant[h // 2 - overlap:, :]
        t_txt, t_conf = self._ocr_image(_upscale_min_height(top, 64))
        b_txt, b_conf = self._ocr_image(_upscale_min_height(bot, 64))
        if not t_txt and not b_txt:
            return "", 0.0
        confs = [c for c, t in ((t_conf, t_txt), (b_conf, b_txt)) if t]
        return t_txt + b_txt, float(np.mean(confs))

    def _read_plate_text(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Multi-variant OCR. Variant 1 is the original pipeline (so cars behave exactly as before);
        extra variants only run when it did not already yield a confident valid plate.
        """
        try:
            h, w = plate_crop.shape[:2]
            two_line = (w / float(max(h, 1))) < 2.3

            v1 = preprocess_plate_crop(plate_crop)
            candidates: List[Tuple[str, float]] = []

            txt, conf = self._ocr_image(v1)
            candidates.append((txt, conf))
            if extract_valid_plate(txt) and conf >= 0.5:
                return txt, conf

            v2 = _plain_variant(plate_crop)
            candidates.append(self._ocr_image(v2))
            if two_line:
                candidates.append(self._read_two_line(v2))
                candidates.append(self._read_two_line(v1 if v1.shape[0] >= 64 else _upscale_min_height(v1, 64)))

            valid = [(t, c) for t, c in candidates if t and extract_valid_plate(t)]
            if valid:
                return max(valid, key=lambda tc: tc[1])
            usable = [(t, c) for t, c in candidates if len(t) >= 4]
            if usable:
                return max(usable, key=lambda tc: (len(tc[0]), tc[1]))
            return None, 0.0
        except Exception:
            return None, 0.0

    def _aggregate_trajectory(self, track_id: int) -> Tuple[Optional[str], float]:
        """Weighted Levenshtein temporal voting (single high-confidence reads allowed, kept provisional)."""
        observations = self.track_observations.get(track_id, [])
        if not observations:
            return None, 0.0

        single_conf = float(getattr(config, "PLATE_ACCEPT_SINGLE_CONF", 0.45))
        clusters = defaultdict(list)
        for obs in observations:
            corrected = extract_valid_plate(obs['raw_text'])
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
        best_support = 0

        for centroid, item_list in clusters.items():
            cluster_score = 0.0
            candidate_counts = defaultdict(float)
            candidate_confs = defaultdict(list)

            for string_val, obs in item_list:
                # Formula: Weight = DetConf * OCRConf * sqrt(BBoxArea)
                weight = obs['det_conf'] * obs['ocr_conf'] * (obs['bbox_area'] ** 0.5)
                cluster_score += weight
                candidate_counts[string_val] += weight
                candidate_confs[string_val].append(obs['ocr_conf'])

            top_ocr = max(o['ocr_conf'] for _, o in item_list)
            qualifies = len(item_list) >= 2 or top_ocr >= single_conf
            if qualifies and cluster_score > max_score:
                max_score = cluster_score
                best_plate = max(candidate_counts, key=candidate_counts.get)
                confs = candidate_confs[best_plate]
                best_conf = float(np.mean(confs)) if confs else 0.85
                best_support = len(item_list)

        if best_plate:
            if best_support >= 2:
                self.provisional.discard(track_id)
            else:
                self.provisional.add(track_id)
        return best_plate, round(best_conf, 2)

    def needs_ocr(self, track_id: int) -> bool:
        with self._lock:
            existing = self.results.get(track_id)
            if existing and (STD_PLATE_REGEX.match(existing) or BH_PLATE_REGEX.match(existing)):
                # a plate backed by one read keeps getting confirmed (up to the attempt cap)
                if track_id not in self.provisional:
                    return False
            if self.attempts.get(track_id, 0) >= 15:
                return False
            return True

    def reset(self) -> None:
        """Clear all per-session state (track IDs restart on every new video)."""
        with self._lock:
            self.results.clear()
            self.plate_boxes.clear()
            self.plate_confidences.clear()
            self.pending_futures.clear()
            self.attempts.clear()
            self.track_observations.clear()
            self.provisional.clear()
            self._frame_counter = 0

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