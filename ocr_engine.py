"""OCR Engine — DISABLED. Plate detection and EasyOCR are commented out.
Only vehicle detection runs. Re-enable by uncommenting when plate model is ready."""

from __future__ import annotations

import threading
from typing import Dict, Optional

# ── Everything below is commented out until plate model is available ──────
# import re
# from concurrent.futures import Future, ThreadPoolExecutor
# from typing import Set
# import cv2
# import easyocr
# import numpy as np
# from detector import PlateDetector

import config


def is_valid_plate(text: str) -> bool:
    return config.MIN_PLATE_CHARS <= len(text) <= config.MAX_PLATE_CHARS


class OCREngine:
    """
    STUB — All plate detection and OCR is disabled.
    Returns empty results so the rest of the pipeline works normally.
    Replace this file with the original ocr_engine.py when plate model is ready.
    """

    def __init__(self, device: str, max_workers: int = config.OCR_MAX_WORKERS):
        self.device = device
        self._lock = threading.Lock()
        # No PlateDetector, no EasyOCR reader, no thread pool
        print("[OCREngine] STUB mode — plate detection disabled. Vehicle detection only.")

    def needs_ocr(self, track_id: int) -> bool:
        return False  # Never run OCR

    def get_plate(self, track_id: int) -> Optional[str]:
        return None

    def get_all_plates(self) -> Dict[int, str]:
        return {}  # Always empty

    @property
    def total_plates_detected(self) -> int:
        return 0

    def submit_vehicle_crop(self, track_id: int, vehicle_crop) -> None:
        pass  # Do nothing

    def drain_completed(self) -> None:
        pass  # Nothing to drain

    def shutdown(self) -> None:
        pass  # Nothing to shut down