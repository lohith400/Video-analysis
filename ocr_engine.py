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
        self.pending_futures: Dict[int, Future[Optional[str]]] = {}  # track_id -> Future
        self.attempts: Dict[int, int] = {}  # track_id -> number of attempts
        print(f"[OCREngine] Initialized on {device} with {max_workers} thread pool workers.")

    def needs_ocr(self, track_id: int) -> bool:
        """Determines if a vehicle track requires OCR."""
        with self._lock:
            if track_id in self.results:
                return False
            if track_id in self.pending_futures:
                return False
            # Max out at 3 attempts per vehicle to conserve GPU/CPU resources
            if self.attempts.get(track_id, 0) >= 3:
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

    def _process_crop(self, track_id: int, vehicle_crop: np.ndarray) -> Optional[str]:
        """Runs inside the thread pool: detects plate crop and applies OCR."""
        try:
            # 1. Extract plate bounding box
            plate_crop = self.detector.detect_plate_region(vehicle_crop)
            if plate_crop is None or plate_crop.size == 0:
                return None

            # 2. Run EasyOCR on the plate crop
            results = self.reader.readtext(plate_crop)
            if not results:
                return None

            # 3. Join detected words, keep alphanumeric only and convert to uppercase
            text = "".join([res[1] for res in results])
            text = re.sub(r'[^A-Za-z0-9]', '', text).upper()

            if is_valid_plate(text):
                return text
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
                            self.results[track_id] = result
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

    @property
    def total_plates_detected(self) -> int:
        """Returns the total count of successfully detected and read license plates."""
        with self._lock:
            return len(self.results)

    def shutdown(self) -> None:
        """Shuts down the background thread pool executor."""
        print("[OCREngine] Shutting down thread pool executor...")
        self.executor.shutdown(wait=False)