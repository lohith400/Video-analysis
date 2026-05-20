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
) -> np.ndarray:
    out = frame.copy()
    
    # 1. Draw the virtual counting gate
    if counter and counter.line_coords:
        line_start, line_end = counter.line_coords
        if counter.was_crossing_recent():
            line_color = (0, 255, 0)       # Flashes thick green on crossing
            thickness = 4
        else:
            line_color = (0, 140, 255)     # Premium Orange/Cyan sleek line
            thickness = 2
            
        cv2.line(out, line_start, line_end, line_color, thickness, cv2.LINE_AA)
        
        # Add gate label
        mid_x = (line_start[0] + line_end[0]) // 2
        mid_y = (line_start[1] + line_end[1]) // 2
        cv2.putText(
            out,
            "COUNTING GATE",
            (mid_x - 60, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            line_color,
            1,
            cv2.LINE_AA,
        )

    # 2. Draw vehicle bounding boxes and labels
    for v in vehicles:
        # Determine color from raw or mapped class
        color = config.BOX_COLORS.get(v.vehicle_class, (200, 200, 200))
        x1, y1, x2, y2 = v.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Use voted class name if counter is available for stability
        label_class = counter.get_voted_class(v.track_id) if counter else v.vehicle_class
        label_text = f"{label_class} ID:{v.track_id}"
        
        _draw_label(out, label_text, x1, y1, color)
        plate = plate_texts.get(v.track_id)
        if plate:
            _draw_label_below(out, plate, x1, y1, color)

    # 3. Draw heads-up display (HUD)
    _draw_hud(out, category_counts, fps, total_plates, counter)
    return out


def _draw_label(
    frame: np.ndarray, text: str, x: int, y: int, color: tuple
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    ty = max(y - 6, th + 4)
    _draw_text_bg(frame, text, x, ty, tw, th, baseline, font, scale, thickness, color)


def _draw_label_below(
    frame: np.ndarray, text: str, x: int, y: int, color: tuple
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    ty = y + th + 10
    _draw_text_bg(frame, text, x, ty, tw, th, baseline, font, scale, thickness, color)


def _draw_text_bg(
    frame, text, x, ty, tw, th, baseline, font, scale, thickness, color
) -> None:
    cv2.rectangle(
        frame,
        (x, ty - th - 4),
        (x + tw + 4, ty + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x + 2, ty), font, scale, color, thickness, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    counts: Dict[str, int],
    fps: float,
    total_plates: int,
    counter: Optional["TrafficCounter"] = None,
) -> None:
    if counter:
        # Use cumulative high-precision line-crossing counts
        cum_counts = counter.get_counts()
        lines = [
            f"FPS: {fps:.1f} | GATE: ACTIVE",
            f"Total Counted: {cum_counts.get('total', 0)}",
            f"Cars: {cum_counts.get('Car', 0)} | Bikes: {cum_counts.get('Bike/Motorcycle', 0)} | Buses: {cum_counts.get('Bus', 0)}",
            f"Trucks: {cum_counts.get('Truck', 0)} | Rickshaws: {cum_counts.get('Auto Rickshaw', 0)}",
            f"Bicycles: {cum_counts.get('Bicycle', 0)} | Vans: {cum_counts.get('Van', 0)} | Others: {cum_counts.get('Others', 0)}",
            f"Plates detected (session): {total_plates}",
        ]
    else:
        # Fallback to legacy raw frame-by-frame counts
        lines = [
            f"FPS: {fps:.1f}",
            f"Total vehicles: {counts.get('total', 0)}",
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
