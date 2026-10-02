"""
Pure-logic plate utilities: positional correction, multi-line character sorting,
weighted Levenshtein temporal clustering, and NMS.
"""

from __future__ import annotations

import collections
import re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import numpy as np

# Valid Indian States and Union Territories
VALID_STATES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CH", "CG", "DD", "DL", "DN", "GA",
    "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN",
    "ML", "MZ", "NL", "OD", "OR", "PY", "PB", "RJ", "SK", "TN", "TS", "TG",
    "TR", "UP", "UK", "UA", "WB",
}

# OCR character confusion mappings
ALPHA_TO_NUM = {
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 
    'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'A': '4', 'T': '7', 'J': '3'
}
NUM_TO_ALPHA = {
    '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', 
    '6': 'G', '7': 'T', '8': 'B', '3': 'E'
}

STD_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BH_PLATE_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}$")


class PositionalRegexCorrector:
    """Applies index-specific character transformations based on Indian plate formats."""

    @staticmethod
    def correct_string(raw_str: str) -> Optional[str]:
        if not raw_str:
            return None
            
        s = re.sub(r'[^A-Z0-9]', '', str(raw_str).upper())
        if len(s) < 8 or len(s) > 11:
            return None

        # Check for Bharat (BH) Series
        if s[2:4] in ['BH', '8H', 'B1', '81']:
            return PositionalRegexCorrector._correct_bh(s)
        return PositionalRegexCorrector._correct_standard(s)

    @staticmethod
    def _correct_standard(s: str) -> Optional[str]:
        s_list = list(s)
        n = len(s_list)

        # Indices 0, 1: State Code (Enforce Alpha)
        s_list[0] = NUM_TO_ALPHA.get(s_list[0], s_list[0])
        s_list[1] = NUM_TO_ALPHA.get(s_list[1], s_list[1])

        # Indices -4 to -1: Unique Identification Number (Enforce Digits)
        for i in range(n - 4, n):
            s_list[i] = ALPHA_TO_NUM.get(s_list[i], s_list[i])

        # Indices 2, 3: RTO District Code (Enforce Digits)
        s_list[2] = ALPHA_TO_NUM.get(s_list[2], s_list[2])
        if s_list[3].isdigit() or s_list[3] in ALPHA_TO_NUM:
            s_list[3] = ALPHA_TO_NUM.get(s_list[3], s_list[3])
            series_start = 4
        else:
            series_start = 3

        # Middle series section (Enforce Alpha)
        for i in range(series_start, n - 4):
            s_list[i] = NUM_TO_ALPHA.get(s_list[i], s_list[i])

        res = "".join(s_list)

        # State code normalization for common misreads
        state = res[:2]
        state_replacements = {
            "MA": "MH", "M4": "MH", "M0": "MH",
            "CQ": "CG", "C0": "CG",
            "K4": "KA", "H4": "KA", "KB": "KA", "K1": "KL",
            "D1": "DL", "OL": "DL", "QL": "DL",
            "T5": "TS", "T0": "TS", "U1": "UP"
        }
        if state in state_replacements:
            res = state_replacements[state] + res[2:]

        if res[:2] in VALID_STATES and STD_PLATE_REGEX.match(res):
            return res
        return None

    @staticmethod
    def _correct_bh(s: str) -> Optional[str]:
        s_list = list(s)
        n = len(s_list)

        # Indices 0, 1: Year Prefix (Enforce Digits)
        s_list[0] = ALPHA_TO_NUM.get(s_list[0], s_list[0])
        s_list[1] = ALPHA_TO_NUM.get(s_list[1], s_list[1])

        # Indices 2, 3: Static BH Token
        s_list[2], s_list[3] = 'B', 'H'

        # Indices 4 to 7: Random Unique Number (Enforce Digits)
        for i in range(4, min(8, n)):
            s_list[i] = ALPHA_TO_NUM.get(s_list[i], s_list[i])

        # Tail Sequence: Series Letters (Enforce Alpha excluding I and O)
        for i in range(8, n):
            s_list[i] = NUM_TO_ALPHA.get(s_list[i], s_list[i])
            if s_list[i] in ['I', 'O']:
                s_list[i] = 'A'

        res = "".join(s_list)
        if BH_PLATE_REGEX.match(res):
            return res
        return None


def extract_valid_plate(raw_str: str) -> Optional[str]:
    """
    Recover a valid Indian plate from a noisy OCR string.

    Tries the whole string first (identical to the old behaviour), then slides a
    window of 10 -> 8 characters across it. This rescues reads polluted by stray
    characters ("IND", bolt holes, stickers, neighbouring text) that previously made
    the whole read be thrown away because of the strict 8-11 length gate.
    """
    if not raw_str:
        return None
    s = re.sub(r"[^A-Z0-9]", "", str(raw_str).upper())
    direct = PositionalRegexCorrector.correct_string(s)
    if direct:
        return direct
    if len(s) <= 8:
        return None
    for win in (10, 9, 8):
        if len(s) < win:
            continue
        for start in range(0, len(s) - win + 1):
            cand = PositionalRegexCorrector.correct_string(s[start:start + win])
            if cand:
                return cand
    return None


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def sort_character_boxes_multiline(char_boxes: list) -> list:
    """
    Sorts OCR character boxes for multi-line license plates (e.g., Two-wheelers / Commercial).
    Clusters boxes vertically by centroid height, then sorts left-to-right.
    """
    if not char_boxes:
        return []

    boxes_with_meta = []
    for box in char_boxes:
        # box structure: [bbox, text, conf] where bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        pts = box[0]
        x_min = min(p[0] for p in pts)
        x_max = max(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        y_max = max(p[1] for p in pts)
        
        cy = (y_min + y_max) / 2.0
        h = y_max - y_min
        boxes_with_meta.append({'box': box, 'cy': cy, 'h': h, 'x1': x_min})

    # Sort primarily by vertical Y-axis
    boxes_with_meta.sort(key=lambda item: item['cy'])

    lines = []
    current_line = [boxes_with_meta[0]]

    for item in boxes_with_meta[1:]:
        mean_h = np.mean([b['h'] for b in current_line])
        if abs(item['cy'] - current_line[-1]['cy']) < (mean_h * 0.5):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    # Sort each row horizontally left-to-right
    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda item: item['x1'])
        sorted_boxes.extend([item['box'] for item in line])

    return sorted_boxes


def clean_and_correct_indian_plate(text: str) -> Optional[str]:
    return PositionalRegexCorrector.correct_string(text)


def best_plate_cluster(history: Sequence[str]) -> Tuple[Optional[str], int, int]:
    """Fallback cluster resolver for simple sequences."""
    if not history:
        return None, 0, 0

    valid_hits = []
    for t in history:
        corrected = clean_and_correct_indian_plate(t)
        if corrected:
            valid_hits.append(corrected)

    if valid_hits:
        winner, count = collections.Counter(valid_hits).most_common(1)[0]
        return winner, count, len(history)

    pool = [t for t in history if len(t) >= 6]
    if not pool:
        return None, 0, len(history)

    winner, count = collections.Counter(pool).most_common(1)[0]
    corrected = clean_and_correct_indian_plate(winner)
    return corrected if corrected else winner, count, len(history)


def nms(
    boxes: Sequence[Tuple[float, float, float, float]],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
) -> List[int]:
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes)
    scores_arr = np.array(scores)

    x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_arr.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep