"""Per-track plate evidence: keep the best native-resolution plate crops seen over the WHOLE track.

The old flow read plates while they were still small and stopped as soon as two reads agreed.
Here every detected plate crop is stored (best K by quality, spread over time) and only read at
the end, together, by plate_fusion.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Tuple

import cv2
import numpy as np

MAX_CROPS = 14          # per track
MIN_HEIGHT = 22         # px; smaller plates carry no usable detail
MIN_FRAME_GAP = 2       # frames between kept crops (avoid 14 copies of the same instant)


def quality(crop: np.ndarray, det_conf: float) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = g.shape[:2]
    s = 160.0 / max(h, 1)
    gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    sharp = min(float(cv2.Laplacian(gs, cv2.CV_64F).var()), 600.0) / 600.0 + 0.15
    return float(h) ** 1.5 * sharp * (0.4 + det_conf)


class PlateEvidence:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[int, List[dict]] = {}

    def add(self, track_id: int, frame_idx: int, frame: np.ndarray, box: Tuple[int, int, int, int],
            det_conf: float) -> None:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bh < MIN_HEIGHT or bw < 20:
            return
        H, W = frame.shape[:2]
        px, py = max(4, int(bw * 0.06)), max(4, int(bh * 0.10))
        crop = frame[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)].copy()
        if crop.size == 0:
            return
        rec = {"q": quality(crop, det_conf), "crop": crop, "frame": frame_idx, "box": box, "det": det_conf}
        with self._lock:
            lst = self._store.setdefault(track_id, [])
            near = [r for r in lst if abs(r["frame"] - frame_idx) < MIN_FRAME_GAP]
            if near:
                if rec["q"] > near[0]["q"]:
                    lst[lst.index(near[0])] = rec
                return
            if len(lst) < MAX_CROPS:
                lst.append(rec)
                return
            worst = min(lst, key=lambda r: r["q"])
            if rec["q"] > worst["q"] * 1.05:
                lst[lst.index(worst)] = rec

    def get(self, track_id: int) -> List[dict]:
        with self._lock:
            return sorted(self._store.get(track_id, []), key=lambda r: r["q"], reverse=True)

    def pop(self, track_id: int) -> List[dict]:
        with self._lock:
            return sorted(self._store.pop(track_id, []), key=lambda r: r["q"], reverse=True)

    def count(self, track_id: int) -> int:
        with self._lock:
            return len(self._store.get(track_id, []))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
