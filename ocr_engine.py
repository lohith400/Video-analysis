"""OCR Engine — Active and fully functional.
Performs license plate region extraction and EasyOCR text extraction asynchronously."""

from __future__ import annotations

import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Set
import cv2
import easyocr
import numpy as np
from detector import PlateDetector

import config


# Valid Indian State Codes
VALID_STATES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", 
    "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ", 
    "NL", "OD", "OR", "PY", "PB", "RJ", "SK", "TN", "TS", "TG", "TR", "UP", "UK", 
    "UA", "WB"
}

# OCR character confusion mappings
CONFUSED_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T', '4': 'A', '3': 'E'
}
CONFUSED_TO_DIGIT = {
    'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2', 'A': '4', 'G': '6', 'T': '7', 'Q': '0', 'D': '0', 'E': '3'
}


def preprocess_plate_image(plate_crop: np.ndarray) -> np.ndarray:
    """Applies advanced preprocessing to enhance license plate text visibility for EasyOCR."""
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop
        
    h, w = plate_crop.shape[:2]
    
    # 1. Upscale plate if it's too small (height < 150)
    if h < 150:
        scale = 150.0 / h
        new_w = int(w * scale)
        plate_crop = cv2.resize(plate_crop, (new_w, 150), interpolation=cv2.INTER_CUBIC)
        
    # 2. Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE to boost contrast locally
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 4. Bilateral filter to smooth flat areas and keep text edges sharp
    processed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return processed


def clean_and_correct_indian_plate(text: str) -> Optional[str]:
    """Cleans and applies position-based character correction heuristics for Indian license plates."""
    # Clean characters: uppercase alphanumeric only
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    
    length = len(text)
    if length < config.MIN_PLATE_CHARS or length > config.MAX_PLATE_CHARS:
        return None
        
    def to_let(c): return CONFUSED_TO_LETTER.get(c, c)
    def to_dig(c): return CONFUSED_TO_DIGIT.get(c, c)
    
    chars = list(text)
    
    # Position-based corrections for Indian plate formats
    if length >= 7:
        # Standard: XX NN ... NNNN
        # 1. First 2 characters must be letters (State Code)
        chars[0] = to_let(chars[0])
        chars[1] = to_let(chars[1])
        
        # 2. Next 1 or 2 characters must be digits (RTO Code)
        if length >= 8:
            chars[2] = to_dig(chars[2])
            chars[3] = to_dig(chars[3])
            
            # Middle characters (from index 4 up to len-4) are letters (Series)
            for i in range(4, length - 4):
                chars[i] = to_let(chars[i])
                
            # Last 4 characters are digits (Unique Number)
            for i in range(length - 4, length):
                chars[i] = to_dig(chars[i])
        else:
            # Length 7: e.g., XX N NNNN or XX NN NNN
            chars[2] = to_dig(chars[2])
            # Last 4 are digits
            for i in range(3, 7):
                chars[i] = to_dig(chars[i])
                
    elif length >= 4:
        # Very short plates: make first letters and last digits
        has_letters = any(c.isalpha() for c in chars)
        if has_letters:
            chars[0] = to_let(chars[0])
            for i in range(1, length):
                chars[i] = to_dig(chars[i])
                
    corrected_text = "".join(chars)
    
    # Strict validation of State Code: every valid plate must start with a valid state code
    if len(corrected_text) >= 2:
        # Apply state corrections first
        state = corrected_text[:2]
        # Direct correction of common state OCR errors (e.g. M4 -> MH, MA -> MH, CQ -> CG)
        if state in ("MA", "M4", "M0"):
            corrected_text = "MH" + corrected_text[2:]
        elif state == "CQ":
            corrected_text = "CG" + corrected_text[2:]
        elif state in ("K4", "H4"):
            corrected_text = "KA" + corrected_text[2:]
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


class OCREngine:
    """Runs license plate detection and EasyOCR asynchronously in a worker thread pool."""

    def __init__(self, device: str, max_workers: int = config.OCR_MAX_WORKERS):
        self.device = device
        self._lock = threading.Lock()
        self.detector = PlateDetector(device)
        self.reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results: Dict[int, str] = {}  # track_id -> plate text
        self.plate_boxes: Dict[int, Tuple[int, int, int, int]] = {}  # track_id -> relative bbox (px1, py1, px2, py2)
        self.pending_futures: Dict[int, Future[Optional[Tuple[str, Tuple[int, int, int, int]]]]] = {}  # track_id -> Future
        self.attempts: Dict[int, int] = {}  # track_id -> number of attempts
        print(f"[OCREngine] Initialized on {device} with {max_workers} thread pool workers.")

    def needs_ocr(self, track_id: int) -> bool:
        """Determines if a vehicle track requires OCR."""
        with self._lock:
            if track_id in self.results:
                return False
            if track_id in self.pending_futures:
                return False
            # Max out at 10 attempts per vehicle to conserve GPU/CPU resources
            if self.attempts.get(track_id, 0) >= 10:
                return False
            return True

    def submit_vehicle_crop(self, track_id: int, vehicle_crop: np.ndarray) -> None:
        """Submits a vehicle crop for plate extraction and EasyOCR in the thread pool."""
        if not self.needs_ocr(track_id):
            return

        with self._lock:
            self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
            future = self.executor.submit(self._process_crop, track_id, vehicle_crop)
            self.pending_futures[track_id] = future

    def _process_crop(self, track_id: int, vehicle_crop: np.ndarray) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
        """Runs inside the thread pool: detects plate crop, preprocesses it, and applies OCR."""
        try:
            # 1. Extract plate bounding box and crop
            detect_res = self.detector.detect_plate_bbox(vehicle_crop)
            if detect_res is None:
                return None
            (px1, py1, px2, py2), plate_crop = detect_res

            # Apply advanced preprocessing to make characters highly visible
            processed_plate = preprocess_plate_image(plate_crop)

            # 2. Run EasyOCR on the preprocessed plate crop with restrict allowlist
            results = self.reader.readtext(
                processed_plate,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            if not results:
                return None

            # 3. Join detected words and apply Indian plate correction heuristics
            text = "".join([res[1] for res in results])
            corrected = clean_and_correct_indian_plate(text)
            if corrected:
                return corrected, (px1, py1, px2, py2)
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
                            plate_text, bbox = result
                            self.results[track_id] = plate_text
                            self.plate_boxes[track_id] = bbox
                    except Exception as exc:
                        print(f"[OCREngine] Future error for track {track_id}: {exc}")
                    completed_ids.append(track_id)

            for track_id in completed_ids:
                self.pending_futures.pop(track_id, None)

    def get_plate(self, track_id: int) -> Optional[str]:
        """Gets the detected plate text for a specific track ID."""
        with self._lock:
            return self.results.get(track_id)

    def get_all_plates(self) -> Dict[int, str]:
        """Gets a snapshot mapping of all track IDs to their detected plate text."""
        with self._lock:
            return dict(self.results)

    def get_plate_box(self, track_id: int) -> Optional[Tuple[int, int, int, int]]:
        """Gets the detected plate relative bounding box for a track ID."""
        with self._lock:
            return self.plate_boxes.get(track_id)

    def get_all_plate_boxes(self) -> Dict[int, Tuple[int, int, int, int]]:
        """Gets a snapshot mapping of all track IDs to their detected plate bounding box."""
        with self._lock:
            return dict(self.plate_boxes)

    @property
    def total_plates_detected(self) -> int:
        """Returns the total count of successfully detected and read license plates."""
        with self._lock:
            return len(self.results)

    def shutdown(self) -> None:
        """Shuts down the background thread pool executor."""
        print("[OCREngine] Shutting down thread pool executor...")
        self.executor.shutdown(wait=False)