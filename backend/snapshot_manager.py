"""Best-snapshot-per-vehicle: keeps ONE highest-quality crop per track_id."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# Expansion of the detector box so rider heads / pillions / plates are not cut off
_EXPAND_TW = (0.15, 0.95, 0.15, 0.06)    # left, top, right, bottom (two-wheelers)
_EXPAND_DEFAULT = (0.06, 0.06, 0.06, 0.06)
MAX_ALTS = 3          # best candidates kept per vehicle (extra ones only used as backup/voting)
MIN_GAP = 6           # frames between kept candidates
EVAL_EVERY = 2        # score every N frames (4K crops are expensive)


def expanded_crop(frame: np.ndarray, bbox, two_wheeler: bool) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns (crop, (origin_x, origin_y))."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    l, t, r, b = _EXPAND_TW if two_wheeler else _EXPAND_DEFAULT
    cx1, cy1 = max(0, int(x1 - bw * l)), max(0, int(y1 - bh * t))
    cx2, cy2 = min(fw, int(x2 + bw * r)), min(fh, int(y2 + bh * b))
    return frame[cy1:cy2, cx1:cx2].copy(), (cx1, cy1)


def _sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 256.0 / max(g.shape)
    if s < 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class SnapshotManager:
    def __init__(self, lost_after_frames: int = 45, min_height: int = 40):
        self.lost_after = lost_after_frames
        self.min_height = min_height
        self.best: Dict[int, list] = {}
        self.last_seen: Dict[int, int] = {}

    def update(self, frame: np.ndarray, vehicles: List, frame_idx: int) -> None:
        fh, fw = frame.shape[:2]
        for v in vehicles:
            x1, y1, x2, y2 = v.bbox
            bw, bh = x2 - x1, y2 - y1
            self.last_seen[v.track_id] = frame_idx
            if bh < self.min_height or bw < 8 or (frame_idx + v.track_id) % EVAL_EVERY:
                continue
            clipped = x1 <= 2 or y1 <= 2 or x2 >= fw - 2 or y2 >= fh - 2
            tw = config.is_two_wheeler(v.vehicle_class)
            crop, origin = expanded_crop(frame, v.bbox, tw)
            if crop.size == 0:
                continue
            sharp = min(_sharpness(crop), 400.0) / 400.0 + 0.2
            score = float(np.sqrt(bw * bh)) * sharp * (0.5 if clipped else 1.0) * (0.5 + v.confidence)
            rec = {"score": score, "crop": crop, "origin": origin, "bbox": v.bbox,
                   "frame_idx": frame_idx, "raw_class": v.vehicle_class, "two_wheeler": tw}
            lst = self.best.setdefault(v.track_id, [])
            near = [r for r in lst if abs(r["frame_idx"] - frame_idx) < MIN_GAP]
            if near:                       # same moment: keep the better one
                if score > near[0]["score"] * 1.08:
                    lst[lst.index(near[0])] = rec
            elif len(lst) < MAX_ALTS:
                lst.append(rec)
            else:
                worst = min(lst, key=lambda r: r["score"])
                if score > worst["score"] * 1.08:
                    lst[lst.index(worst)] = rec

    def pop_lost(self, frame_idx: int) -> List[Tuple[int, dict]]:
        lost = [t for t, f in self.last_seen.items() if frame_idx - f > self.lost_after]
        return self._pop(lost)

    def pop_all(self) -> List[Tuple[int, dict]]:
        return self._pop(list(self.last_seen.keys()))

    def _pop(self, tids) -> List[Tuple[int, dict]]:
        out = []
        for t in tids:
            self.last_seen.pop(t, None)
            lst = self.best.pop(t, None)
            if lst:
                lst = sorted(lst, key=lambda r: r["score"], reverse=True)
                best = dict(lst[0])
                best["alts"] = lst[1:]
                out.append((t, best))
        return out