"""OCR Engine — Active and fully functional.
Performs license plate region extraction and EasyOCR text extraction asynchronously."""

from __future__ import annotations

import collections
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Tuple, List
import cv2
import easyocr
import numpy as np
from detector import PlateDetector

import config
from plate_utils import (
    INDIAN_PLATE_REGEX,
    best_plate_cluster,
    clean_and_correct_indian_plate,
    nms,
)


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


def preprocess_vehicle_crop_for_detection(vehicle_crop: np.ndarray) -> np.ndarray:
    """Sharpens vehicle crop prior to plate detection: Grayscale -> CLAHE 2.0 -> BGR."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return vehicle_crop
    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_plate_for_ocr(plate_crop: np.ndarray) -> np.ndarray:
    """Applies advanced preprocessing: upscale to min 200px, grayscale, CLAHE 3.0, bilateral filter, Otsu threshold."""
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop
        
    h, w = plate_crop.shape[:2]
    # 1. Upscale plate crop to minimum 200px height (keep aspect ratio)
    if h < 200:
        scale = 200.0 / h
        new_w = int(w * scale)
        plate_crop = cv2.resize(plate_crop, (new_w, 200), interpolation=cv2.INTER_CUBIC)
        
    # 2. Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE (clipLimit=3.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 4. Apply cv2.bilateralFilter to remove noise while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 5. Apply Otsu thresholding
    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded


class OCREngine:
    """Runs license plate detection and EasyOCR asynchronously in a worker thread pool."""

    def __init__(self, device: str, max_workers: int = config.OCR_MAX_WORKERS):
        self.device = device
        self._lock = threading.Lock()
        self.detector = PlateDetector(device)
        self.reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # State tracking
        self.results: Dict[int, str] = {}  # track_id -> plate text
        self.plate_boxes: Dict[int, Tuple[int, int, int, int]] = {}  # track_id -> ABSOLUTE frame bbox (x1, y1, x2, y2)
        self.pending_futures: Dict[int, Future[Optional[Tuple[str, Tuple[int, int, int, int]]]]] = {}  # track_id -> Future
        self.attempts: Dict[int, int] = {}  # track_id -> number of attempts
        
        # rolling buffers for stabilization and voting
        self.vehicle_crop_buffer: Dict[int, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = {}  # track_id -> [(crop, bbox), ...]
        self.ocr_history: Dict[int, List[str]] = {}  # track_id -> list of valid OCR texts (up to 10)
        
        print(f"[OCREngine] Initialized on {device} with {max_workers} thread pool workers.")

    def needs_ocr(self, track_id: int) -> bool:
        """Determines if a vehicle track requires more OCR attempts."""
        with self._lock:
            # Only stop if a fully-valid plate (matches Indian regex) is already confirmed
            existing = self.results.get(track_id)
            if existing and INDIAN_PLATE_REGEX.match(existing) is not None:
                return False  # We have a valid plate — no more OCR needed
            if track_id in self.pending_futures:
                return False  # Already running
            # Allow up to 10 attempts per vehicle
            if self.attempts.get(track_id, 0) >= 10:
                return False
            return True

    def submit_vehicle_crop(self, track_id: int, vehicle_crop: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> None:
        """Submits a rolling window of 3 vehicle crops to the thread pool for stabilization and OCR."""
        if not self.needs_ocr(track_id):
            return

        with self._lock:
            if track_id not in self.vehicle_crop_buffer:
                self.vehicle_crop_buffer[track_id] = []
            self.vehicle_crop_buffer[track_id].append((vehicle_crop.copy(), vehicle_bbox))
            
            # Keep buffer size at max 3
            if len(self.vehicle_crop_buffer[track_id]) > 3:
                self.vehicle_crop_buffer[track_id].pop(0)

            # Only run stabilization and OCR when we have exactly 3 consecutive crops
            if len(self.vehicle_crop_buffer[track_id]) == 3:
                self.attempts[track_id] = self.attempts.get(track_id, 0) + 1
                crops_to_process = list(self.vehicle_crop_buffer[track_id])
                future = self.executor.submit(self._process_crops_stabilized, track_id, crops_to_process)
                self.pending_futures[track_id] = future

    def _process_crops_stabilized(
        self, track_id: int, crops_to_process: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
    ) -> Optional[Tuple[str, Tuple[int, int, int, int], bool]]:
        """Runs inside the thread pool: stabilizes plate detection across 3 frames, preprocesses and applies OCR with fallbacks."""
        try:
            all_plates = []  # List of tuples: (abs_bbox, confidence, plate_crop, vehicle_bbox)
            
            for vehicle_crop, (vx1, vy1, vx2, vy2) in crops_to_process:
                # 1. Preprocess the vehicle crop for edge sharpness
                preprocessed_vehicle = preprocess_vehicle_crop_for_detection(vehicle_crop)
                
                # 2. Run plate detector on the preprocessed crop
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
                    # Skip if confidence is less than PLATE_CONF_THRESHOLD
                    if conf < config.PLATE_CONF_THRESHOLD:
                        continue
                        
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    h, w = vehicle_crop.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    # Calculate absolute box in frame coordinates
                    abs_x1 = vx1 + x1
                    abs_y1 = vy1 + y1
                    abs_x2 = vx1 + x2
                    abs_y2 = vy1 + y2
                    
                    plate_crop = vehicle_crop[y1:y2, x1:x2].copy()
                    all_plates.append(((abs_x1, abs_y1, abs_x2, abs_y2), conf, plate_crop, (vx1, vy1, vx2, vy2)))
                    
            if not all_plates:
                return None
                
            # Run NMS to merge overlapping boxes across all 3 frames
            bboxes = [p[0] for p in all_plates]
            confs = [p[1] for p in all_plates]
            
            keep_indices = nms(bboxes, confs, iou_threshold=0.5)
            if not keep_indices:
                return None
                
            # Keep the highest confidence detection
            best_idx = keep_indices[0]
            best_abs_bbox, best_conf, best_plate_crop, (vx1, vy1, vx2, vy2) = all_plates[best_idx]

            # IMPORTANT: keep this as an ABSOLUTE frame-coordinate box.
            #
            # This detection ran inside a background thread, on a vehicle
            # crop captured 1-3 frames ago (vx1, vy1 = that OLD vehicle
            # position). By the time this result is drawn, the caller
            # (annotator.py) only knows the vehicle's CURRENT bbox, which
            # has moved since. Previously this method converted the box to
            # be *relative to the old vehicle position* (vx1, vy1), and
            # annotator.py re-added the *current* vehicle position on top
            # of that — silently mixing two different points in time and
            # causing the green box to drift away from the actual plate.
            #
            # Storing/drawing the absolute position directly removes that
            # double coordinate transform. The box is still up to a few
            # frames old, but it's drawn where the plate actually was,
            # not offset by however far the vehicle has since moved.
            abs_bbox = best_abs_bbox

            # Apply advanced binarization OCR preprocessing
            thresholded_plate = preprocess_plate_for_ocr(best_plate_crop)

            # Run EasyOCR with detail=0, paragraph=False on standard binarized plate
            results = self.reader.readtext(
                thresholded_plate,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=0,
                paragraph=False
            )
            raw_text = "".join(results)
            raw_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

            # Apply position-based Indian plate corrections first
            corrected_text = clean_and_correct_indian_plate(raw_text)
            
            is_valid = False
            text = raw_text
            if corrected_text is not None:
                # Check valid plate regex on corrected standard output
                is_valid = INDIAN_PLATE_REGEX.match(corrected_text) is not None
                text = corrected_text
                
            if not is_valid:
                # Try reading again with the inverted image
                inverted_plate = cv2.bitwise_not(thresholded_plate)
                results_inv = self.reader.readtext(
                    inverted_plate,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    detail=0,
                    paragraph=False
                )
                raw_text_inv = "".join(results_inv)
                raw_text_inv = re.sub(r'[^A-Z0-9]', '', raw_text_inv.upper())
                
                corrected_text_inv = clean_and_correct_indian_plate(raw_text_inv)
                if corrected_text_inv is not None:
                    # Check valid plate regex on corrected inverted output
                    if INDIAN_PLATE_REGEX.match(corrected_text_inv) is not None:
                        text = corrected_text_inv
                        is_valid = True
                    elif not text:  # Fallback to corrected inverted text if standard text is empty
                        text = corrected_text_inv
                elif not text:
                    text = raw_text_inv

            # Return text, relative bounding box, and whether the text matches standard Indian format
            return text, abs_bbox, is_valid
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
                            plate_text, abs_bbox, is_valid = result

                            # Store the ABSOLUTE frame-coordinate box directly (see note
                            # in _process_crops_stabilized) so annotator.py can draw it
                            # as-is without re-basing it onto the vehicle's current bbox.
                            self.plate_boxes[track_id] = abs_bbox
                            
                            # Print a debug message to monitor OCR progress
                            print(f"[OCREngine] Track #{track_id} OCR read: '{plate_text}' (is_valid: {is_valid})")
                            
                            # Require at least 2 matching OCR reads before accepting a plate.
                            # This filters single-attempt garbage reads like 'J5Y5RLG' or 'O57L'.
                            if plate_text and config.MIN_PLATE_CHARS <= len(plate_text) <= config.MAX_PLATE_CHARS:
                                # 1. Update rolling OCR history for clustered voting (up to 10 reads)
                                if track_id not in self.ocr_history:
                                    self.ocr_history[track_id] = []
                                self.ocr_history[track_id].append(plate_text)
                                if len(self.ocr_history[track_id]) > 10:
                                    self.ocr_history[track_id].pop(0)

                                # 2. Group near-identical reads (edit-distance <= 2) instead of
                                #    requiring an exact string match. This stops frame-to-frame
                                #    OCR noise ('KA05NH8088' vs 'PKA05NH8088' vs 'AO5H88088')
                                #    from splitting votes across near-duplicates.
                                history = self.ocr_history[track_id]
                                winner, count, total = best_plate_cluster(history)
                                confidence = (count / total) * 100 if total else 0.0

                                # 3. Validity is checked against the WINNING candidate, not the
                                #    single latest raw read — a run of noisy reads shouldn't be
                                #    labeled "VALID" just because the newest one happened to parse.
                                winner_corrected = clean_and_correct_indian_plate(winner) if winner else None
                                winner_is_valid = bool(
                                    winner_corrected and INDIAN_PLATE_REGEX.match(winner_corrected)
                                )
                                display_text = winner_corrected if winner_corrected else winner

                                # 4. Log to console
                                import time
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                valid_tag = "✓ VALID" if winner_is_valid else "~ PARTIAL"
                                print(f"[{timestamp}] Vehicle #{track_id} → Plate: {display_text} ({valid_tag}, votes: {count}/{total}, conf: {confidence:.0f}%)")

                                # 5. MINIMUM 2 VOTES (within the winning cluster) required before
                                #    saving to results — blocks single-attempt garbage.
                                if count >= 2:
                                    existing = self.results.get(track_id)
                                    existing_valid = existing and INDIAN_PLATE_REGEX.match(existing) is not None
                                    # Prefer valid plates; upgrade a partial to a better partial
                                    # only if the new candidate has more votes backing it.
                                    if not existing or (winner_is_valid and not existing_valid) or (
                                        not existing_valid and count > history.count(existing)
                                    ):
                                        self.results[track_id] = display_text
                                    print(f"[OCREngine] ✓ Plate confirmed for track #{track_id}: {display_text}")
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