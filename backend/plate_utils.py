"""
Pure-logic plate utilities: character correction, format validation, NMS.

Split out of ocr_engine.py deliberately. ocr_engine.py imports cv2 and
easyocr at module level (the latter also lazily pulls in torch on first
use), which makes it slow and heavy to import just to test string-cleaning
logic. None of the functions below touch an image, a model, or a GPU —
they're plain string/array math, so they belong in their own module that
CI and unit tests can import in milliseconds.
"""

from __future__ import annotations

import collections
import re
from typing import List, Optional, Sequence, Tuple

import config

# Valid Indian State Codes
VALID_STATES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CH", "CG", "DD", "DL", "DN", "GA",
    "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN",
    "ML", "MZ", "NL", "OD", "OR", "PY", "PB", "RJ", "SK", "TN", "TS", "TG",
    "TR", "UP", "UK", "UA", "WB",
}

# OCR character confusion mappings
CONFUSED_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T",
    "4": "A", "3": "E",
}
CONFUSED_TO_DIGIT = {
    "O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "A": "4",
    "G": "6", "T": "7", "Q": "0", "D": "0", "E": "3",
}

# Indian plate regex format checker: SS NN LL NNNN (1-2 letter series)
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")


def clean_and_correct_indian_plate(text: str) -> Optional[str]:
    """Cleans and applies position-based character correction heuristics
    for Indian license plates. Returns None if the result can't be
    corrected into a plausible plate (wrong length or invalid state code).
    """
    text = re.sub(r"[^A-Za-z0-9]", "", text).upper()

    length = len(text)
    if length < config.MIN_PLATE_CHARS or length > config.MAX_PLATE_CHARS:
        return None

    def to_let(c: str) -> str:
        return CONFUSED_TO_LETTER.get(c, c)

    def to_dig(c: str) -> str:
        return CONFUSED_TO_DIGIT.get(c, c)

    chars = list(text)

    if length >= 7:
        # Standard: XX NN ... NNNN
        chars[0] = to_let(chars[0])
        chars[1] = to_let(chars[1])

        if length >= 8:
            chars[2] = to_dig(chars[2])
            chars[3] = to_dig(chars[3])
            for i in range(4, length - 4):
                chars[i] = to_let(chars[i])
            for i in range(length - 4, length):
                chars[i] = to_dig(chars[i])
        else:
            # Length 7: e.g., XX N NNNN or XX NN NNN
            chars[2] = to_dig(chars[2])
            for i in range(3, 7):
                chars[i] = to_dig(chars[i])

    elif length >= 4:
        has_letters = any(c.isalpha() for c in chars)
        if has_letters:
            chars[0] = to_let(chars[0])
            for i in range(1, length):
                chars[i] = to_dig(chars[i])

    corrected_text = "".join(chars)

    if len(corrected_text) >= 2:
        state = corrected_text[:2]
        if state in ("MA", "M4", "M0"):
            corrected_text = "MH" + corrected_text[2:]
        elif state == "CQ":
            corrected_text = "CG" + corrected_text[2:]
        elif state in ("K4", "H4"):
            corrected_text = "KA" + corrected_text[2:]
        elif state == "D1":
            corrected_text = "DL" + corrected_text[2:]
        elif state in ("T5", "T0"):
            corrected_text = "TS" + corrected_text[2:]
        elif state == "U1":
            corrected_text = "UP" + corrected_text[2:]

        final_state = corrected_text[:2]
        if final_state not in VALID_STATES:
            return None
    else:
        return None

    return corrected_text


def is_valid_plate(text: str) -> bool:
    return config.MIN_PLATE_CHARS <= len(text) <= config.MAX_PLATE_CHARS


def edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance. Small strings (plates are <=10 chars)
    so the O(n*m) DP table here is cheap — no need for a library."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def plates_agree(a: str, b: str, max_edit: int = 2) -> bool:
    """Two OCR reads are considered the same plate if they're close enough
    (small edit distance) rather than requiring an exact string match.
    This absorbs the frame-to-frame OCR noise (dropped/extra/confused
    characters) that otherwise splits votes across near-identical reads
    and lets an early garbage read 'win' a majority vote forever."""
    return edit_distance(a, b) <= max_edit


def best_plate_cluster(history: Sequence[str]) -> Tuple[Optional[str], int, int]:
    """Groups a rolling history of raw OCR reads into edit-distance clusters
    and returns (representative_text, votes_in_winning_cluster, total_reads).

    representative_text is the *longest* member of the winning cluster
    (more characters generally means a cleaner read), not just whichever
    string happened to appear first — this avoids getting stuck on a short
    early garbage read (e.g. 'XU86') once better, longer reads for the same
    plate start showing up (e.g. 'KA05NH8088').
    """
    if not history:
        return None, 0, 0

    # Priority 1: if ANY read in history already corrects into a fully valid
    # Indian plate, trust that over raw clustering. Short garbage reads
    # ('YI', 'TF', 'FIE') are dangerous to cluster by edit distance — a
    # 2-char edit budget is huge relative to a 2-4 char string, so unrelated
    # noise can coincidentally cluster together and outvote a real plate
    # that only appeared twice. A validated, correctly-formatted plate is a
    # much stronger signal than cluster size alone.
    valid_hits = []
    for text in history:
        corrected = clean_and_correct_indian_plate(text)
        if corrected and INDIAN_PLATE_REGEX.match(corrected):
            valid_hits.append(corrected)

    if valid_hits:
        winner, _ = collections.Counter(valid_hits).most_common(1)[0]
        votes = sum(
            1 for t in history
            if clean_and_correct_indian_plate(t) == winner or plates_agree(t, winner)
        )
        return winner, votes, len(history)

    # Priority 2: nothing validated yet. Cluster by edit distance, but only
    # among reads long enough to plausibly be a real (partial) plate, so
    # short noise strings don't dominate the pool.
    pool = [t for t in history if len(t) >= config.MIN_PLATE_CHARS] or list(history)

    clusters: List[List[str]] = []
    for text in pool:
        placed = False
        for cluster in clusters:
            if plates_agree(text, cluster[0]):
                cluster.append(text)
                placed = True
                break
        if not placed:
            clusters.append([text])

    best_cluster = max(clusters, key=len)

    # Picking the longest raw string as the representative is a trap: a
    # single stray extra character ('PKA05NH8088' from a smudge/reflection)
    # will beat the true, clean 'KA05NH8088' just by being longer. Instead
    # use the most common LENGTH within the cluster (repeated reads tend to
    # agree on the true plate length even when individual characters are
    # noisy), then the most frequent raw string at that length.
    lengths = collections.Counter(len(c) for c in best_cluster)
    modal_len = lengths.most_common(1)[0][0]
    same_len = [c for c in best_cluster if len(c) == modal_len]
    cluster_counts = collections.Counter(best_cluster)
    representative = max(same_len, key=lambda c: cluster_counts[c])

    return representative, len(best_cluster), len(history)


def nms(
    boxes: Sequence[Tuple[float, float, float, float]],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
) -> List[int]:
    """Pure Python/NumPy Non-Maximum Suppression to merge overlapping
    plate bounding boxes. Returns indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    import numpy as np  # local import: keeps module import itself numpy-free

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
