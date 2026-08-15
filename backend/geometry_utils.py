"""
Pure-logic geometry helpers shared by the pedestrian pipeline.

Same rationale as plate_utils.py / helmet_logic.py / schemas.py: IoU math
and box filtering don't need a YOLO model in scope, so they're split out
of pedestrian_detector.py (which imports ultralytics/torch at module
level) into a module CI and unit tests can import instantly.
"""

from __future__ import annotations

from typing import Tuple, TypedDict


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

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_child_by_height(height: float, avg_adult_height: float, ratio: float) -> bool:
    """Scene-relative child heuristic: shorter than `ratio` of the current
    scene's average adult height gets reclassified as Child, regardless
    of the raw classifier output. Documented limitation: this is a
    deliberate compensation for a very small demographics training set,
    not a learned category — see LIMITATIONS.md.
    """
    if avg_adult_height <= 0.0:
        return False
    return height < ratio * avg_adult_height
