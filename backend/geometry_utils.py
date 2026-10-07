"""
Pure-logic geometry and temporal smoothing helpers for the pedestrian pipeline.
IoU math, rider exclusion, temporal confidence-weighted voting, and rolling median
smoothing are kept here so they can be imported and unit-tested without PyTorch/YOLO.
"""

from __future__ import annotations

import collections
import statistics
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict


class HasBBox(TypedDict):
    bbox: Tuple[int, int, int, int]
    track_id: int


def calculate_iou(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> float:
    """Intersection-over-Union of two (x1, y1, x2, y2) boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_intersection_ratio(
    box_inner: Tuple[int, int, int, int], box_outer: Tuple[int, int, int, int]
) -> float:
    """
    Computes intersection(box_inner, box_outer) / area(box_inner).
    Used to detect if a pedestrian crop is largely contained within a vehicle box.
    """
    inter_x1 = max(box_inner[0], box_outer[0])
    inter_y1 = max(box_inner[1], box_outer[1])
    inter_x2 = min(box_inner[2], box_outer[2])
    inter_y2 = min(box_inner[3], box_outer[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_inner = max(0, box_inner[2] - box_inner[0]) * max(0, box_inner[3] - box_inner[1])
    return inter_area / area_inner if area_inner > 0 else 0.0


def is_center_inside_expanded_box(
    inner_box: Tuple[int, int, int, int],
    outer_box: Tuple[int, int, int, int],
    upward_expansion_ratio: float = 0.40,
) -> bool:
    """
    Checks if the center point of inner_box lies within outer_box expanded vertically
    upward by `upward_expansion_ratio` * height.
    Captures riders and pillions whose torso/head extends above a motorcycle/scooter/auto.
    """
    cx = (inner_box[0] + inner_box[2]) / 2.0
    cy = (inner_box[1] + inner_box[3]) / 2.0

    ox1, oy1, ox2, oy2 = outer_box
    vh = max(0, oy2 - oy1)
    expanded_oy1 = oy1 - (upward_expansion_ratio * vh)

    return (ox1 <= cx <= ox2) and (expanded_oy1 <= cy <= oy2)


def is_rider_of_vehicle(
    person_box: Tuple[int, int, int, int],
    vehicle_box: Tuple[int, int, int, int],
    vehicle_class: str = "",
    overlap_thresh: float = 0.50,
    upward_expansion_ratio: float = 0.40,
) -> bool:
    """
    Determines if a detected person is actually a vehicle rider/pillion.
    Criteria:
      1. intersection(person, vehicle) / person_area >= overlap_thresh (default 0.50)
      OR
      2. person center is inside the vehicle box expanded upward by 40% (for two-wheelers, bicycles, autos).
    """
    # 1. Direct overlap ratio test
    overlap_ratio = calculate_intersection_ratio(person_box, vehicle_box)
    if overlap_ratio >= overlap_thresh:
        return True

    # 2. Check if two-wheeler / auto / bicycle center containment applies
    v_cls = str(vehicle_class).lower().strip()
    is_two_wheeler_or_small = any(
        kw in v_cls
        for kw in ["bike", "motorcycle", "scooter", "two-wheeler", "bicycle", "auto", "rickshaw", "three-wheeler"]
    ) or (not v_cls)  # if class unspecified, perform general check

    if is_two_wheeler_or_small:
        if is_center_inside_expanded_box(person_box, vehicle_box, upward_expansion_ratio):
            return True

    return False


def is_child_by_height(height: float, avg_adult_height: float, ratio: float = 0.60) -> bool:
    """
    Weak scene-relative child height heuristic.
    Should only serve as an optional tie-breaker or fallback; never overrides high-confidence model output.
    """
    if avg_adult_height <= 0.0:
        return False
    return height < ratio * avg_adult_height


def resolve_demographic_label(
    model_label: str,
    model_conf: float,
    height: float = 0.0,
    avg_adult_height: float = 0.0,
    use_height_heuristic: bool = False,
    child_height_ratio: float = 0.60,
    conf_threshold: float = 0.50,
) -> Tuple[str, float]:
    """
    Determines final demographic label strictly respecting:
      - Model prediction takes priority.
      - Child is assigned only if model predicts child with conf >= threshold, OR
        weak height heuristic is explicitly enabled AND model is uncertain.
      - Never silently converts unknown to male.
    """
    lbl = model_label.lower().strip()
    if model_conf < conf_threshold:
        # If model is uncertain and height heuristic is enabled, allow weak hint
        if use_height_heuristic and is_child_by_height(height, avg_adult_height, child_height_ratio):
            return "child", 0.51
        return "unknown", model_conf

    # High-confidence model prediction wins unconditionally
    if any(w in lbl for w in ["child", "kid"]):
        return "child", model_conf
    elif any(w in lbl for w in ["female", "woman", "girl"]):
        return "female_adult", model_conf
    elif any(w in lbl for w in ["male", "man", "boy"]):
        return "male_adult", model_conf
    else:
        return "unknown", model_conf


class TrackVoteAccumulator:
    """
    Per-track temporal confidence-weighted vote history.
    Prevents frame-to-frame flip-flopping across demographic labels.
    """

    def __init__(self, window_size: int = 15, min_votes: int = 3, weight_threshold: float = 0.60):
        self.window_size = window_size
        self.min_votes = min_votes
        self.weight_threshold = weight_threshold
        # track_id -> deque of (label, confidence)
        self.history: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.window_size)
        )

    def add_vote(self, track_id: int, label: str, conf: float) -> None:
        if label and label != "unknown":
            self.history[track_id].append((label, conf))

    def get_voted_label(self, track_id: int) -> Tuple[str, float]:
        votes = self.history.get(track_id)
        if not votes or len(votes) < self.min_votes:
            return "unknown", 0.0

        weights: Dict[str, float] = collections.defaultdict(float)
        total_weight = 0.0

        for lbl, conf in votes:
            weights[lbl] += conf
            total_weight += conf

        if total_weight <= 0.0:
            return "unknown", 0.0

        winner, win_weight = max(weights.items(), key=lambda item: item[1])
        win_ratio = win_weight / total_weight

        if win_ratio >= self.weight_threshold:
            return winner, win_ratio
        return "unknown", win_ratio

    def clear_track(self, track_id: int) -> None:
        self.history.pop(track_id, None)


class RollingMedianSmoother:
    """
    Maintains a time-based rolling median buffer to eliminate snapshot count spikes.
    """

    def __init__(self, window_duration_sec: float = 2.0):
        self.window_duration_sec = window_duration_sec
        # list of (timestamp_seconds, count)
        self.buffer: collections.deque = collections.deque()

    def update(self, current_time: float, count: int) -> int:
        self.buffer.append((current_time, count))
        cutoff = current_time - self.window_duration_sec

        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

        if not self.buffer:
            return count

        counts = [c for _, c in self.buffer]
        return int(statistics.median(counts))
