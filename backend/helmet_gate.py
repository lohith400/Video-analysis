"""Geometry gate for helmet heads: a head must sit on top of a person, not on a wheel or a seat.

Measured on L3 with the real helmet model: wheel / bag / seat / image-edge boxes scored higher
(0.54, 0.31, 0.66) than the genuine, small heads (0.17), so the old "top 2 by confidence" choice
picked the wrong boxes. Gating first, ranking second, fixes that without touching the model.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

EDGE = 0.012            # fraction of image size treated as "touching the border"
HEAD_TOP_FRAC = 0.38    # head centre must be in the top 38% of its person box
HEAD_H_RANGE = (0.07, 0.40)   # head height / person height


def touches_edge(b: Box, W: int, H: int) -> Tuple[bool, bool]:
    """(rejecting_edge, clipped_top). Left/right/bottom contact rejects; top contact only flags."""
    ex, ey = EDGE * W, EDGE * H
    bad = b[0] <= ex or b[2] >= W - ex or b[3] >= H - ey
    return bad, b[1] <= ey


def match_person(head: Box, persons: Sequence[Box]) -> Optional[int]:
    """Index of the person whose upper body contains this head, else None."""
    cx, cy = (head[0] + head[2]) / 2.0, (head[1] + head[3]) / 2.0
    hh = head[3] - head[1]
    best, best_i = None, None
    for i, p in enumerate(persons):
        pw, ph = p[2] - p[0], p[3] - p[1]
        if ph <= 0 or not (p[0] - 0.1 * pw <= cx <= p[2] + 0.1 * pw):
            continue
        if not (p[1] - 0.15 * ph <= cy <= p[1] + HEAD_TOP_FRAC * ph):
            continue
        r = hh / ph
        if not (HEAD_H_RANGE[0] <= r <= HEAD_H_RANGE[1]):
            continue
        d = abs(cy - p[1])
        if best is None or d < best:
            best, best_i = d, i
    return best_i


def filter_persons(persons: Sequence[Box], vehicle_box: Optional[Box]) -> List[Box]:
    """Keep only people on THIS vehicle: inside the vehicle box, or (no box) the biggest person
    plus anyone overlapping that person (pillion). Drops pedestrians in the background."""
    if not persons:
        return []
    def area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    def inter(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if vehicle_box is not None:
        vb = (vehicle_box[0] - 0.1 * (vehicle_box[2] - vehicle_box[0]), vehicle_box[1] - 0.15 * (vehicle_box[3] - vehicle_box[1]),
              vehicle_box[2] + 0.1 * (vehicle_box[2] - vehicle_box[0]), vehicle_box[3])
        kept = [p for p in persons if area(p) > 0 and inter(p, vb) / area(p) >= 0.45]
        if kept:
            return kept
    main = max(persons, key=area)
    return [p for p in persons if p is main or (area(p) > 0 and inter(p, main) / area(p) >= 0.15)]


def gate(clusters: List[dict], W: int, H: int, persons: Sequence[Box],
         vehicle_box: Optional[Box]) -> List[dict]:
    """Keep plausible heads. Each cluster dict needs 'box'. Adds 'person' and 'clipped_top'."""
    persons = filter_persons(persons, vehicle_box)
    kept: List[dict] = []
    for c in clusters:
        b = c["box"]
        if (b[2] - b[0]) * (b[3] - b[1]) > 0.30 * W * H:
            continue
        bad, clipped = touches_edge(b, W, H)
        if bad:
            continue
        if persons:
            pi = match_person(b, persons)
            if pi is None:
                continue
            c["person"] = pi
        elif vehicle_box is not None:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            bw, bh = vehicle_box[2] - vehicle_box[0], vehicle_box[3] - vehicle_box[1]
            if not (vehicle_box[0] - 0.2 * bw <= cx <= vehicle_box[2] + 0.2 * bw) or cy > vehicle_box[1] + 0.40 * bh:
                continue
            c["person"] = None
        c["clipped_top"] = clipped
        kept.append(c)
    # one head per person: the most confident
    best = {}
    out = []
    for c in sorted(kept, key=lambda c: c["conf"], reverse=True):
        key = c.get("person")
        if key is None:
            out.append(c)
        elif key not in best:
            best[key] = c
            out.append(c)
    return out
