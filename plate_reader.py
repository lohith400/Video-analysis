"""
plate_reader.py — License plate detection + OCR
Supports Indian number plates (white/yellow, regional fonts)
"""

import cv2
import re
import numpy as np
import easyocr
from ultralytics import YOLO


# Indian plate pattern: MH12AB1234 or MH-12-AB-1234
INDIAN_PLATE_PATTERN = re.compile(
    r'[A-Z]{2}[\s\-]?[0-9]{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?[0-9]{1,4}'
)


def preprocess_plate(img):
    """Enhance plate image for OCR accuracy."""
    # Upscale 2x
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


class PlateReader:
    def __init__(self, plate_model_path: str):
        print(f"  Loading plate model: {plate_model_path}")
        try:
            self.detector = YOLO(plate_model_path)
        except Exception:
            print("  ⚠️  Plate model not found — using full-frame OCR fallback")
            self.detector = None

        self.reader = easyocr.Reader(['en'], gpu=False)
        self._cache = {}        # track_id → plate_text

    def _ocr(self, img) -> str:
        processed = preprocess_plate(img)
        results = self.reader.readtext(processed, detail=0,
                                       allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        raw = ''.join(results).upper().replace(' ', '').replace('-', '')
        return raw

    def _validate(self, text: str) -> str | None:
        """Return cleaned plate text if it matches Indian pattern."""
        match = INDIAN_PLATE_PATTERN.search(text)
        return match.group(0).replace(' ', '').replace('-', '') if match else None

    def read(self, vehicle_crop) -> str | None:
        """Detect plate region, then OCR it. Returns plate string or None."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        plate_img = vehicle_crop

        # Use YOLO to find plate region if model is available
        if self.detector is not None:
            results = self.detector(vehicle_crop, verbose=False)[0]
            if results.boxes:
                best = max(results.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                plate_img = vehicle_crop[y1:y2, x1:x2]

        if plate_img.size == 0:
            return None

        raw_text = self._ocr(plate_img)
        return self._validate(raw_text)
