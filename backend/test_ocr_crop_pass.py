"""Logic tests for the two-wheeler/auto crop pass, windowed plate recovery and provisional voting.
Heavy deps (easyocr, ultralytics) are stubbed, so this runs anywhere."""
from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

# ---- stubs injected before importing ocr_engine ----
_state = {"full": [], "crop": [], "ocr": []}


class _T:  # tiny tensor-like
    def __init__(self, a): self.a = np.array(a, dtype=float)
    def cpu(self): return self
    def numpy(self): return self.a


def _res(boxes):
    if not boxes:
        return [SimpleNamespace(boxes=None)]
    return [SimpleNamespace(boxes=SimpleNamespace(
        xyxy=_T([b[:4] for b in boxes]), conf=_T([b[4] for b in boxes]), __len__=None))]


class _Boxes:
    def __init__(self, boxes):
        self.xyxy = _T([b[:4] for b in boxes]); self.conf = _T([b[4] for b in boxes]); self.n = len(boxes)
    def __len__(self): return self.n


class _Model:
    def predict(self, img, imgsz=None, **kw):
        h, w = img.shape[:2]
        src = _state["full"] if (w > 1500 or imgsz == 1280) else _state["crop"]
        return [SimpleNamespace(boxes=_Boxes(src) if src else None)]


class _PD:
    def __init__(self, device): self.model = _Model()


sys.modules["detector"] = types.SimpleNamespace(PlateDetector=_PD)


class _Reader:
    def __init__(self, *a, **k): pass
    def readtext(self, img, **k):
        if _state["ocr"]:
            t, c = _state["ocr"].pop(0)
            return [([[0, 0], [10, 0], [10, 10], [0, 10]], t, c)] if t else []
        return []


sys.modules["easyocr"] = types.SimpleNamespace(Reader=_Reader)

import config  # noqa: E402
from ocr_engine import OCREngine  # noqa: E402
from plate_utils import extract_valid_plate  # noqa: E402


def V(tid, cls, bbox):
    return SimpleNamespace(track_id=tid, vehicle_class=cls, bbox=bbox)


class TestExtract(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(extract_valid_plate("KA05NF9274"), "KA05NF9274")

    def test_noise_prefix_suffix(self):
        self.assertEqual(extract_valid_plate("INDKA05NF9274X"), "KA05NF9274")

    def test_garbage(self):
        self.assertIsNone(extract_valid_plate("1K42"))


class TestCropPass(unittest.TestCase):
    def setUp(self):
        _state.update(full=[], crop=[], ocr=[])
        self.e = OCREngine("cpu")
        config.PLATE_CROP_EVERY_N_FRAMES = 1
        self.frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    def test_crop_box_mapped_to_absolute_coords(self):
        # scooter bbox 1000..1400 x 800..1300 (400x500) -> crop upscaled x1.28 (long side 500 < 640)
        v = V(7, "motorcycle", (1000, 800, 1400, 1300))
        scale = 640 / 540.0  # padded crop long side = 500*1.16 = 580 -> compute exactly below
        ch = int(1300 + 500 * 0.08) - int(800 - 500 * 0.08)
        cw = int(1400 + 400 * 0.08) - int(1000 - 400 * 0.08)
        scale = 640 / float(max(ch, cw))
        cx1, cy1 = int(1000 - 400 * 0.08), int(800 - 500 * 0.08)
        # plate at frame coords (1150..1250, 850..900) -> crop coords
        fx1, fy1, fx2, fy2 = 1150, 850, 1250, 900
        _state["crop"] = [((fx1 - cx1) * scale, (fy1 - cy1) * scale, (fx2 - cx1) * scale, (fy2 - cy1) * scale, 0.5)]
        _state["ocr"] = [("KA05NF9274", 0.8)]
        self.e.detect_plates_sync(self.frame, [v])
        bx = self.e.get_plate_box(7)
        self.assertIsNotNone(bx)
        for got, want in zip(bx, (fx1, fy1, fx2, fy2)):
            self.assertLessEqual(abs(got - want), 2)
        self.assertEqual(self.e.get_plate(7), "KA05NF9274")  # single confident read accepted

    def test_single_read_is_provisional_then_confirmed(self):
        v = V(9, "auto-rickshaw", (500, 500, 900, 900))
        _state["crop"] = [(100, 100, 220, 160, 0.6)]
        _state["ocr"] = [("KA05NF9274", 0.6)]
        self.e.detect_plates_sync(self.frame, [v])
        self.assertIn(9, self.e.provisional)
        self.assertTrue(self.e.needs_ocr(9))  # keeps confirming
        _state["ocr"] = [("KA05NF9274", 0.7)]
        self.e.detect_plates_sync(self.frame, [v])
        self.assertNotIn(9, self.e.provisional)
        self.assertFalse(self.e.needs_ocr(9))

    def test_low_conf_single_read_not_published(self):
        v = V(11, "motorcycle", (500, 500, 900, 900))
        _state["crop"] = [(100, 100, 220, 160, 0.6)]
        _state["ocr"] = [("KA05NF9274", 0.2)]
        self.e.detect_plates_sync(self.frame, [v])
        self.assertIsNone(self.e.get_plate(11))

    def test_cars_do_not_use_crop_pass(self):
        v = V(3, "car", (500, 500, 1500, 1200))
        _state["crop"] = [(100, 100, 400, 200, 0.9)]
        _state["ocr"] = [("KA05NF9274", 0.9)]
        self.e.detect_plates_sync(self.frame, [v])
        self.assertIsNone(self.e.get_plate(3))

    def test_neighbour_plate_not_assigned(self):
        v = V(5, "motorcycle", (1000, 800, 1400, 1300))
        # plate far to the left of the vehicle -> centre outside the vehicle bbox
        cx1 = int(1000 - 400 * 0.08)
        _state["crop"] = [(0, 100, 20, 120, 0.9)]
        _state["ocr"] = [("KA05NF9274", 0.9)]
        self.e.detect_plates_sync(self.frame, [v])
        self.assertIsNone(self.e.get_plate_box(5))

    def test_reset(self):
        v = V(9, "auto-rickshaw", (500, 500, 900, 900))
        _state["crop"] = [(100, 100, 220, 160, 0.6)]
        _state["ocr"] = [("KA05NF9274", 0.6)]
        self.e.detect_plates_sync(self.frame, [v])
        self.e.reset()
        self.assertEqual(self.e.get_all_plates(), {})
        self.assertEqual(len(self.e.track_observations), 0)


if __name__ == "__main__":
    unittest.main()
