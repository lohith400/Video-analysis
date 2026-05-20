"""TrafficCounter class using line-crossing geometry, class-voting and track-age filters."""

from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional, Set, Tuple

import config
from tracker import TrackedVehicle


class TrafficCounter:
    """Handles line-crossing mathematics, track age filtering, and class voting/smoothing."""

    def __init__(
        self,
        line_pct: Tuple[float, float, float, float] = config.COUNTING_LINE_PCT,
        class_mapping: Dict[str, str] = config.USER_CLASS_MAPPING,
        min_track_age: int = config.MIN_TRACK_AGE,
    ):
        self.line_pct = line_pct
        self.class_mapping = class_mapping
        self.min_track_age = min_track_age

        # State dictionaries
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}  # track_id -> list of bottom-center (cx, y2)
        self.track_classes: Dict[int, List[str]] = {}             # track_id -> list of detected raw classes
        self.counted_ids: Set[int] = set()                       # Set of track_ids that have crossed the line

        # Initialize counts for each unique target class + total
        self.target_classes = sorted(list(set(class_mapping.values())))
        if "Others" not in self.target_classes:
            self.target_classes.append("Others")
        self.counts: Dict[str, int] = {cls: 0 for cls in self.target_classes}
        self.counts["total"] = 0

        # Frame geometry (resolved dynamically on first frame)
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        # Flash trigger when a vehicle crosses the line
        self.last_crossing_time: float = 0.0

    def _resolve_line_coords(self, width: int, height: int) -> None:
        """Calculate absolute pixel coordinates for the virtual counting line."""
        self.frame_width = width
        self.frame_height = height
        x1_pct, y1_pct, x2_pct, y2_pct = self.line_pct
        A = (int(x1_pct * width), int(y1_pct * height))
        B = (int(x2_pct * width), int(y2_pct * height))
        self.line_coords = (A, B)

    @staticmethod
    def _ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> float:
        """Determines orientation of ordered triplet (A, B, C) for intersection check."""
        return (C[1] - A[1]) * (B[0] - A[0]) - (C[0] - A[0]) * (B[1] - A[1])

    @classmethod
    def check_intersection(
        cls, A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]
    ) -> bool:
        """Returns True if line segment AB intersects line segment CD."""
        return (cls._ccw(A, B, C) > 0) != (cls._ccw(A, B, D) > 0) and (
            cls._ccw(C, D, A) > 0
        ) != (cls._ccw(C, D, B) > 0)

    def get_voted_class(self, track_id: int) -> str:
        """Gets the majority-voted (mode) human-readable class name for a track."""
        raw_classes = self.track_classes.get(track_id)
        if not raw_classes:
            return "Others"
        
        # Count frequencies
        counter = collections.Counter(raw_classes)
        most_common_raw, _ = counter.most_common(1)[0]
        
        # Map raw YOLO class to target class name
        return self.class_mapping.get(most_common_raw, "Others")

    def update(self, vehicles: List[TrackedVehicle], frame_shape: Tuple[int, ...]) -> None:
        """Update tracker history, run class voting, and check for line-crossings."""
        if not vehicles:
            return

        height, width = frame_shape[:2]
        if self.frame_width != width or self.frame_height != height:
            self._resolve_line_coords(width, height)

        assert self.line_coords is not None
        line_start, line_end = self.line_coords

        active_ids = set()
        for v in vehicles:
            track_id = v.track_id
            active_ids.add(track_id)

            # 1. Update raw class prediction history
            if track_id not in self.track_classes:
                self.track_classes[track_id] = []
            self.track_classes[track_id].append(v.vehicle_class)

            # 2. Update bottom-center position history: (cx, y2)
            x1, y1, x2, y2 = v.bbox
            cx = (x1 + x2) // 2
            bottom_center = (cx, y2)

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(bottom_center)

            # Limit history length to save memory (we only need the last 2 frames for intersection)
            if len(self.track_history[track_id]) > 5:
                self.track_history[track_id].pop(0)

            # 3. Check for crossing (if eligible and not already counted)
            if track_id not in self.counted_ids:
                history = self.track_history[track_id]
                # Enforce track age filter to prevent ephemeral/spurious noise from triggering a count
                if len(history) >= 2 and len(self.track_classes[track_id]) >= self.min_track_age:
                    prev_pos = history[-2]
                    curr_pos = history[-1]

                    # Segment intersection check
                    if self.check_intersection(line_start, line_end, prev_pos, curr_pos):
                        self.counted_ids.add(track_id)
                        
                        # Determine majority class for this specific vehicle track
                        voted_class = self.get_voted_class(track_id)
                        self.counts[voted_class] = self.counts.get(voted_class, 0) + 1
                        self.counts["total"] += 1
                        
                        # Set flash trigger
                        self.last_crossing_time = time.time()
                        print(f"[TrafficCounter] Track {track_id} crossed! Class={voted_class}. Cumulative total={self.counts['total']}")

        # Cleanup very old track histories to prevent memory leak (if they haven't been active for 300 frames)
        # We can clean up track histories of tracks not present in active_ids after a certain time,
        # but let's keep it simple: keep history size bounded or clean up when tracks are completely lost.
        # Ultralytics track buffers are max 90, so we can clean up any track not seen in active_ids
        # if it hasn't appeared for 120 frames. For now, since short videos are typical, simple dictionary
        # updates are perfectly fine.

    def was_crossing_recent(self) -> bool:
        """Returns True if a line crossing occurred within the last 0.4 seconds."""
        return (time.time() - self.last_crossing_time) < 0.4

    def get_counts(self) -> Dict[str, int]:
        """Returns a snapshot of the current cumulative vehicle counts."""
        return dict(self.counts)

    def reset(self) -> None:
        """Resets all counter states."""
        self.track_history.clear()
        self.track_classes.clear()
        self.counted_ids.clear()
        self.counts = {cls: 0 for cls in self.target_classes}
        self.counts["total"] = 0
        self.last_crossing_time = 0.0
