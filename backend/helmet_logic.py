"""
Pure-logic rider/pillion helmet classification.

Split out of helmet_checker.py's _run_detection method. The detection
call (running the YOLO model on a crop) has to stay in HelmetChecker, but
everything after that — sorting detected heads left-to-right, assigning
rider vs. pillion, mapping detector classes to helmet/no_helmet, and
building violation records — is plain data transformation with no model
or GPU involved. Keeping it here means the rider/pillion heuristic
(documented, deliberate limitation: positional, not angle-aware) can be
unit tested directly instead of only via a full inference call.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict

import config


class DetectedPerson(TypedDict):
    bbox: Tuple[int, int, int, int]
    x_center: float
    class_name: str
    conf: float


def classify_riders(
    detected_persons: List[DetectedPerson],
    track_id: int,
    plate: str,
    vehicle_class: str,
    timestamp: str,
    vehicle_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[List[Dict], Dict]:
    """Assigns rider (leftmost) / pillion (second-leftmost) roles and
    returns (violations, status_record).

    Positional heuristic: leftmost detected head/helmet box is treated as
    the rider, the second as the pillion. This assumes a conventional
    left-right camera angle and will misclassify in unusual angles or
    overtaking scenarios — see LIMITATIONS.md.
    """
    sorted_persons = sorted(detected_persons, key=lambda p: p["x_center"])

    violations: List[Dict] = []
    vx1, vy1, vx2, vy2 = vehicle_bbox if vehicle_bbox else (0, 0, 0, 0)

    rider_helmet = "unknown"
    pillion_helmet = "none"

    if len(sorted_persons) >= 1:
        rider = sorted_persons[0]
        mapped = config.HELMET_CLASS_MAP.get(rider["class_name"], "helmet")
        rider_helmet = mapped
        if mapped == "no_helmet":
            rx1, ry1, rx2, ry2 = rider["bbox"]
            violations.append({
                "track_id": track_id,
                "plate": plate,
                "vehicle_class": vehicle_class,
                "violation_type": "rider_no_helmet",
                "timestamp": timestamp,
                "frame_violation": True,
                "person_bbox": (vx1 + rx1, vy1 + ry1, vx1 + rx2, vy1 + ry2),
            })

    if len(sorted_persons) >= 2:
        pillion = sorted_persons[1]
        mapped = config.HELMET_CLASS_MAP.get(pillion["class_name"], "helmet")
        pillion_helmet = mapped
        if mapped == "no_helmet":
            px1, py1, px2, py2 = pillion["bbox"]
            violations.append({
                "track_id": track_id,
                "plate": plate,
                "vehicle_class": vehicle_class,
                "violation_type": "pillion_no_helmet",
                "timestamp": timestamp,
                "frame_violation": True,
                "person_bbox": (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2),
            })

    status_record = {
        "track_id": track_id,
        "plate": plate,
        "vehicle_class": vehicle_class,
        "rider_helmet": rider_helmet,
        "pillion_helmet": pillion_helmet,
        "timestamp": timestamp,
    }

    return violations, status_record
