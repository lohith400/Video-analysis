"""
TrafficCounter — 2D Vector Cross-Product Geometry Counting Line.
Guarantees zero duplicate counts and detects movement direction without frame-age heuristics.
"""

from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

import config
from schemas import TrackedVehicle


class TrafficCounter:
    def __init__(
        self,
        line_pct: Tuple[float, float, float, float] = config.COUNTING_LINE_PCT,
        class_mapping: Dict[str, str] = config.USER_CLASS_MAPPING,
        min_track_age: int = config.MIN_TRACK_AGE,
    ):
        self.line_pct = line_pct
        self.class_mapping = class_mapping
        self.min_track_age = min_track_age

        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_classes: Dict[int, List[str]] = {}
        self.counted_ids: Set[int] = set()
        self.track_assigned_class: Dict[int, str] = {}
        self.track_directions: Dict[int, str] = {}

        self.target_classes = sorted(list(set(class_mapping.values())))
        if "Others" not in self.target_classes:
            self.target_classes.append("Others")
        self.counts: Dict[str, int] = {cls: 0 for cls in self.target_classes}
        self.counts["total"] = 0

        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.last_crossing_time: float = 0.0

    def _resolve_line_coords(self, width: int, height: int) -> None:
        self.frame_width = width
        self.frame_height = height
        x1_pct, y1_pct, x2_pct, y2_pct = self.line_pct
        self.A = np.array([int(x1_pct * width), int(y1_pct * height)], dtype=np.float32)
        self.B = np.array([int(x2_pct * width), int(y2_pct * height)], dtype=np.float32)
        self.line_coords = ((int(self.A[0]), int(self.A[1])), (int(self.B[0]), int(self.B[1])))

    @staticmethod
    def _ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        """2D Cross-Product Vector Orientation."""
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    def get_voted_class(self, track_id: int) -> str:
        raw_classes = self.track_classes.get(track_id)
        if not raw_classes:
            return "Others"
        counter = collections.Counter(raw_classes)
        most_common_raw, _ = counter.most_common(1)[0]
        return self.class_mapping.get(most_common_raw, "Others")

    def update(self, vehicles: List[TrackedVehicle], frame_shape: Tuple[int, ...]) -> None:
        if not vehicles:
            return

        height, width = frame_shape[:2]
        if self.frame_width != width or self.frame_height != height:
            self._resolve_line_coords(width, height)

        for v in vehicles:
            track_id = v.track_id

            if track_id not in self.track_classes:
                self.track_classes[track_id] = []
            self.track_classes[track_id].append(v.vehicle_class)

            x1, y1, x2, y2 = v.bbox
            cx = (x1 + x2) // 2
            bottom_center = (cx, y2)

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(bottom_center)

            if len(self.track_history[track_id]) > 10:
                self.track_history[track_id].pop(0)

            # Vector-Intersection Line Crossing Test
            if track_id not in self.counted_ids and len(self.track_history[track_id]) >= 2:
                P_prev = np.array(self.track_history[track_id][-2], dtype=np.float32)
                P_curr = np.array(self.track_history[track_id][-1], dtype=np.float32)

                ccw_A_B_Pprev = self._ccw(self.A, self.B, P_prev)
                ccw_A_B_Pcurr = self._ccw(self.A, self.B, P_curr)
                ccw_P_A = self._ccw(P_prev, P_curr, self.A)
                ccw_P_B = self._ccw(P_prev, P_curr, self.B)

                straddle_line = (np.sign(ccw_A_B_Pprev) != np.sign(ccw_A_B_Pcurr)) and (ccw_A_B_Pprev != 0)
                straddle_trajectory = (np.sign(ccw_P_A) != np.sign(ccw_P_B))

                if straddle_line and straddle_trajectory:
                    new_voted_class = self.get_voted_class(track_id)
                    direction = "INBOUND" if ccw_A_B_Pcurr > 0 else "OUTBOUND"
                    
                    self.counted_ids.add(track_id)
                    self.track_assigned_class[track_id] = new_voted_class
                    self.track_directions[track_id] = direction
                    
                    self.counts[new_voted_class] = self.counts.get(new_voted_class, 0) + 1
                    self.counts["total"] += 1
                    self.last_crossing_time = time.time()
                    
                    print(f"[TrafficCounter] Track {track_id} crossed ({direction})! Class={new_voted_class}. Total={self.counts['total']}")

    def was_crossing_recent(self) -> bool:
        return (time.time() - self.last_crossing_time) < 0.4

    def get_counts(self) -> Dict[str, int]:
        return dict(self.counts)

    def reset(self) -> None:
        self.track_history.clear()
        self.track_classes.clear()
        self.counted_ids.clear()
        self.track_assigned_class.clear()
        self.track_directions.clear()
        self.counts = {cls: 0 for cls in self.target_classes}
        self.counts["total"] = 0