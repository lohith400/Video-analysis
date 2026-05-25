"""Frame annotation: vehicle boxes, track/plate labels, HUD overlay."""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

import config
from tracker import TrackedVehicle


def draw_annotations(
    frame: np.ndarray,
    vehicles: List[TrackedVehicle],
    plate_texts: Dict[int, str],
    category_counts: Dict[str, int],
    fps: float,
    total_plates: int,
    counter: Optional["TrafficCounter"] = None,
    plate_boxes: Optional[Dict[int, tuple]] = None,
    violations: Optional[List[Dict]] = None,
    pedestrians: Optional[Dict] = None,
) -> np.ndarray:
    out = frame.copy()

    for v in vehicles:
        color = config.BOX_COLORS.get(v.vehicle_class, (200, 200, 200))
        x1, y1, x2, y2 = v.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label_class = counter.get_voted_class(v.track_id) if counter else v.vehicle_class
        label_text = f"{label_class} ID:{v.track_id}"
        _draw_label(out, label_text, x1, y1, color)

        plate = plate_texts.get(v.track_id)
        if plate:
            _draw_label_below(out, plate, x1, y2, color)

        # FIX: plate_boxes stores coords relative to the PADDED vehicle crop.
        # The padded crop origin was passed as vehicle_bbox in submit_vehicle_crop.
        # Since we now pass padded_bbox (not v.bbox), the rel coords map back to
        # the padded origin — which is already in original frame space.
        # So abs_plate = padded_origin + rel_plate. But we only have v.bbox here,
        # not padded_bbox. The padding is config-side (15px), so we subtract it:
        if plate_boxes and v.track_id in plate_boxes:
            px1, py1, px2, py2 = plate_boxes[v.track_id]
            pad = 15  # must match crop_vehicle() pad parameter
            # padded origin = (x1 - pad, y1 - pad) clamped to 0
            padded_vx1 = max(0, x1 - pad)
            padded_vy1 = max(0, y1 - pad)
            abs_px1 = padded_vx1 + px1
            abs_py1 = padded_vy1 + py1
            abs_px2 = padded_vx1 + px2
            abs_py2 = padded_vy1 + py2
            cv2.rectangle(out, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)

    if violations:
        for viol in violations:
            if "person_bbox" in viol:
                px1, py1, px2, py2 = viol["person_bbox"]
                cv2.rectangle(out, (px1, py1), (px2, py2), (0, 0, 255), 2)
                _draw_label(out, "NO HELMET", px1, py1, (0, 0, 255))
                plate_text = viol.get("plate", "UNKNOWN")
                v_type = viol.get("violation_type", "")
                v_label = v_type.replace("_", " ").title()
                below_text = f"{plate_text} - {v_label}"
                _draw_label_below(out, below_text, px1, py2, (0, 0, 255))

    if pedestrians and "details" in pedestrians:
        for p in pedestrians["details"]:
            if "bbox" in p:
                px1, py1, px2, py2 = p["bbox"]
                gender = p.get("gender", "unknown")
                if gender == "male_adult":
                    color = (255, 0, 0)
                    label = "Male"
                elif gender == "female_adult":
                    color = (128, 0, 128)
                    label = "Female"
                elif gender == "child":
                    color = (0, 255, 255)
                    label = "Child"
                else:
                    color = (255, 255, 255)
                    label = "Person"
                cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)
                _draw_label(out, label, px1, py1, color)

    _draw_hud(out, category_counts, fps, total_plates, counter)
    return out


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    ty = max(y - 6, th + 4)
    _draw_text_bg(frame, text, x, ty, tw, th, baseline, font, scale, thickness, color)


def _draw_label_below(frame: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    ty = y + th + 10
    _draw_text_bg(frame, text, x, ty, tw, th, baseline, font, scale, thickness, color)


def _draw_text_bg(frame, text, x, ty, tw, th, baseline, font, scale, thickness, color) -> None:
    cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, ty), font, scale, color, thickness, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    counts: Dict[str, int],
    fps: float,
    total_plates: int,
    counter: Optional["TrafficCounter"] = None,
) -> None:
    if counter:
        cum_counts = counter.get_counts()
        lines = [
            f"FPS: {fps:.1f} | TRACKING: ACTIVE",
            f"Total Tracked: {cum_counts.get('total', 0)}",
            f"Cars: {cum_counts.get('Car', 0)} | Bikes: {cum_counts.get('Bike/Motorcycle', 0)} | Buses: {cum_counts.get('Bus', 0)}",
            f"Trucks: {cum_counts.get('Truck', 0)} | Rickshaws: {cum_counts.get('Auto Rickshaw', 0)}",
            f"Bicycles: {cum_counts.get('Bicycle', 0)} | Vans: {cum_counts.get('Van', 0)} | Others: {cum_counts.get('Others', 0)}",
            f"Plates detected (session): {total_plates}",
        ]
    else:
        lines = [
            f"FPS: {fps:.1f} | TRACKING: ACTIVE",
            f"Total Tracked: {counts.get('total', 0)}",
            f"Cars: {counts.get('car', 0)} | Trucks: {counts.get('truck', 0)} | Buses: {counts.get('bus', 0)}",
            f"Auto-rickshaw: {counts.get('auto-rickshaw', 0)} | Motorcycle: {counts.get('motorcycle', 0)}",
            f"Scooter: {counts.get('scooter', 0)} | Bicycle: {counts.get('bicycle', 0)}",
            f"Plates detected (session): {total_plates}",
        ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    x, y0 = 10, 24
    line_h = 22
    pad = 6
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        max_w = max(max_w, tw)

    box_h = line_h * len(lines) + pad * 2
    cv2.rectangle(frame, (x - pad, y0 - 18), (x + max_w + pad * 2, y0 - 18 + box_h), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(frame, line, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def count_by_category(vehicles: List[TrackedVehicle]) -> Dict[str, int]:
    counts = {c: 0 for c in config.ALL_VEHICLE_CLASSES}
    for v in vehicles:
        if v.vehicle_class in counts:
            counts[v.vehicle_class] += 1
    counts["total"] = len(vehicles)
    return counts
