"""OCR Engine — Active and fully functional.
Performs license plate region extraction and EasyOCR text extraction asynchronously."""

from __future__ import annotations

import collections
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Set, Tuple, List
import cv2
import easyocr
import numpy as np
from detector import PlateDetector

import config


VALID_STATES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ",
    "NL", "OD", "OR", "PY", "PB", "RJ", "SK", "TN", "TS", "TG", "TR", "UP", "UK",
    "UA", "WB"
}

CONFUSED_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T', '4': 'A', '3': 'E'
}
CONFUSED_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2', 'A': '4', 'G': '6', 'T': '7', 'Q': '0', 'D': '0', 'E': '3'
}


def preprocess_plate_for_ocr(plate_crop: np.ndarray) -> np.ndarray:
    """Applies advanced preprocessing with green plate detection for EV plates."""
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop

    h, w = plate_crop.shape[:2]
    if h < 200:
        scale = 200.0 / h
        new_w = int(w * scale)
        plate_crop = cv2.resize(plate_crop, (new_w, 200), interpolation=cv2.INTER_CUBIC)

    # Detect green plate (Indian EV plates — KA05NM5938 style)
    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
    is_green_plate = cv2.countNonZero(green_mask) > (plate_crop.shape[0] * plate_crop.shape[1] * 0.15)

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)

    if is_green_plate:
        # Adaptive threshold works much better on green plates than Otsu
        return cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded


def clean_and_correct_indian_plate(text: str) -> Optional[str]:
    """Cleans and applies position-based character correction for Indian license plates."""
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()

    length = len(text)
    if length < config.MIN_PLATE_CHARS or length > config.MAX_PLATE_CHARS:
        return None

    def to_let(c): return CONFUSED_TO_LETTER.get(c, c)
    def to_dig(c): return CONFUSED_TO_DIGIT.get(c, c)

    chars = list(text)

    if length >= 7:
        chars[0] = to_let(chars[0])
        chars[1] = to_let(chars[1])
        if length >= 8:
            chars[2] = to_dig(chars[2])
            chars[3] = to_dig(chars[3])
            for i in range(4, length - 4):
                chars[i] = to_let(chars[i])
            for i in range(length - 4, length):
                chars[i] = to_dig(chars[i])
        else:
            chars[2] = to_dig(chars[2])
            for i in range(3, 7):
                chars[i] = to_dig(chars[i])
    elif length >= 4:
        has_letters = any(c.isalpha() for c in chars)
        if has_letters:
            chars[0] = to_let(chars[0])
            for i in range(1, length):
                chars[i] = to_dig(chars[i])

    corrected_text = "".join(chars)

    if len(corrected_text) >= 2:
        state = corrected_text[:2]
        # FIX: H4 correctly maps to HR (Haryana), not KA (Karnataka)
        if state in ("MA", "M4", "M0"):
            corrected_text = "MH" + corrected_text[2:]
        elif state == "CQ":
            corrected_text = "CG" + corrected_text[2:]
        elif state == "K4":
            corrected_text = "KA" + corrected_text[2:]
        elif state == "H4":
            corrected_text = "HR" + corrected_text[2:]
        elif state == "D1":
            corrected_text = "DL" + corrected_text[2:]
        elif state in ("T5", "T0"):
            corrected_text = "TS" + corrected_text[2:]
        elif state == "U1":
            corrected_text = "UP" + corrected_text[2:]

        final_state = corrected_text[:2]
        if final_state not in VALID_STATES:
            print(f"[OCREngine] Filtering out invalid state code: {corrected_text}")
            return None
    else:
        return None

    return corrected_text


def is_valid_plate(text: str) -> bool:
    return config.MIN_PLATE_CHARS <= len(text) <= config.MAX_PLATE_CHARS


INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')


def nms(boxes, scores, iou_threshold=0.5):
    """Pure Python NMS to merge overlapping plate bboxes."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def preprocess_vehicle_crop_for_detection(vehicle_crop: np.ndarray) -> np.ndarray:
    """Sharpens vehicle crop prior to plate detection: Grayscale -> CLAHE 2.0 -> BGR."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return vehicle_crop
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class OCREngine:
    """Runs license plate detection and EasyOCR asynchronously in a worker thread pool."""

    def __init__(self, device: str, max_workers: int = config.OCR_MAX_WORKERS):
        self.device = device
        self._lock = threading.Lock()
        self.detector = PlateDetector(device)
        self.reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.results: Dict[int, str] = {}
        self.plate_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        self.pending_futures: Dict[int, Future] = {}
        self.attempts: Dict[int, int] = {}

        self.vehicle_crop_buffer: Dict[int, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = {}
        self.ocr_history: Dict[int, List[str]] = {}

        print(f"[OCREngine] Initialized on {device} with {max_workers} thread pool workers.")

    def needs_ocr(self, track_id: int) -> bool:
        with self._lock:
            if track_id in self.results:
                return False
            if track_id in self.pending_futures:
                return False
            # FIX: increased from 10 to 25 — fast vehicles need more attempts
            if self.attempts.get(track_id, 0) >= 25:
                return False
            return True

    def submit_vehicle_crop(
        self, track_id: int, vehicle_crop: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> None:
        """Submits rolling window of 3 vehicle crops for stabilized OCR.
        vehicle_bbox must be the padded bbox returned by crop_vehicle() — NOT v.bbox."""
        if not self.needs_ocr(track_id):
            return

        with self._lock:
            if track_id not in self.vehicle_crop_buffer:
                self.vehicle_crop_buffer[track_id] = []
            self.vehicle_crop_buffer[track_id].append((vehicle_crop.copy(), vehicle_bbox))

            if len(self.vehicle_crop_buffer[track_id]) > 3:
                self.vehicle_crop_buffer[track_id].pop(0)

            if len(self.vehicle_crop_buffer[track_id]) == 3:
                self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
                crops_to_process = list(self.vehicle_crop_buffer[track_id])
                future = self.executor.submit(self._process_crops_stabilized, track_id, crops_to_process)
                self.pending_futures[track_id] = future

    def _process_crops_stabilized(
        self, track_id: int, crops_to_process: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
    ) -> Optional[Tuple[str, Tuple[int, int, int, int], bool]]:
        """Stabilizes plate detection across 3 frames, preprocesses and applies OCR with fallbacks."""
        try:
            all_plates = []

            for vehicle_crop, (vx1, vy1, vx2, vy2) in crops_to_process:
                preprocessed_vehicle = preprocess_vehicle_crop_for_detection(vehicle_crop)

                results = self.detector.model.predict(
                    preprocessed_vehicle,
                    conf=config.PLATE_CONF_THRESHOLD,
                    iou=config.IOU_THRESHOLD,
                    half=config.USE_HALF,
                    device=self.device,
                    verbose=False,
                )
                if not results or results[0].boxes is None:
                    continue

                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i].item())
                    if conf < config.PLATE_CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    h, w = vehicle_crop.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Absolute coords in original frame using padded vehicle origin
                    abs_x1 = vx1 + x1
                    abs_y1 = vy1 + y1
                    abs_x2 = vx1 + x2
                    abs_y2 = vy1 + y2

                    plate_crop = vehicle_crop[y1:y2, x1:x2].copy()
                    all_plates.append(((abs_x1, abs_y1, abs_x2, abs_y2), conf, plate_crop, (vx1, vy1, vx2, vy2)))

            if not all_plates:
                return None

            bboxes = [p[0] for p in all_plates]
            confs = [p[1] for p in all_plates]
            keep_indices = nms(bboxes, confs, iou_threshold=0.5)
            if not keep_indices:
                return None

            best_idx = keep_indices[0]
            best_abs_bbox, best_conf, best_plate_crop, (vx1, vy1, vx2, vy2) = all_plates[best_idx]

            # Relative plate bbox inside the padded vehicle crop
            abs_px1, abs_py1, abs_px2, abs_py2 = best_abs_bbox
            rel_bbox = (abs_px1 - vx1, abs_py1 - vy1, abs_px2 - vx1, abs_py2 - vy1)

            # FIX: preprocess_plate_for_ocr now handles green EV plates correctly
            thresholded_plate = preprocess_plate_for_ocr(best_plate_crop)

            results = self.reader.readtext(
                thresholded_plate,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=0,
                paragraph=False
            )
            raw_text = re.sub(r'[^A-Z0-9]', '', "".join(results).upper())
            corrected_text = clean_and_correct_indian_plate(raw_text)

            is_valid = False
            text = raw_text
            if corrected_text is not None:
                is_valid = INDIAN_PLATE_REGEX.match(corrected_text) is not None
                text = corrected_text

            if not is_valid:
                inverted_plate = cv2.bitwise_not(thresholded_plate)
                results_inv = self.reader.readtext(
                    inverted_plate,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    detail=0,
                    paragraph=False
                )
                raw_text_inv = re.sub(r'[^A-Z0-9]', '', "".join(results_inv).upper())
                corrected_text_inv = clean_and_correct_indian_plate(raw_text_inv)
                if corrected_text_inv is not None:
                    if INDIAN_PLATE_REGEX.match(corrected_text_inv) is not None:
                        text = corrected_text_inv
                        is_valid = True
                    elif not text:
                        text = corrected_text_inv
                elif not text:
                    text = raw_text_inv

            return text, rel_bbox, is_valid

        except Exception as exc:
            print(f"[OCREngine] Error processing track {track_id}: {exc}")
        return None

    def drain_completed(self) -> None:
        """Checks and processes all completed background OCR futures."""
        with self._lock:
            completed_ids = []
            for track_id, future in list(self.pending_futures.items()):
                if future.done():
                    try:
                        result = future.result()
                        if result is not None:
                            plate_text, rel_bbox, is_valid = result
                            self.plate_boxes[track_id] = rel_bbox
                            print(f"[OCREngine] Track #{track_id} OCR read: '{plate_text}' (is_valid: {is_valid})")

                            if is_valid:
                                if track_id not in self.ocr_history:
                                    self.ocr_history[track_id] = []
                                self.ocr_history[track_id].append(plate_text)
                                if len(self.ocr_history[track_id]) > 10:
                                    self.ocr_history[track_id].pop(0)

                                counter = collections.Counter(self.ocr_history[track_id])
                                most_common_plate, count = counter.most_common(1)[0]
                                confidence = (count / len(self.ocr_history[track_id])) * 100

                                import time
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                print(f"[{timestamp}] Vehicle #{track_id} → Plate: {most_common_plate} (confidence: {confidence:.0f}%)")

                                self.results[track_id] = most_common_plate
                    except Exception as exc:
                        print(f"[OCREngine] Future error for track {track_id}: {exc}")
                    completed_ids.append(track_id)

            for track_id in completed_ids:
                self.pending_futures.pop(track_id, None)

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
        print("[OCREngine] Shutting down thread pool executor...")
        self.executor.shutdown(wait=False)
