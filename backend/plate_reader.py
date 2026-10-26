"""Robust plate reading on ONE high-quality vehicle snapshot.

Why a separate module: the live OCR path squeezes every plate to 64px height and
accepts the first 'valid-looking' read, which is how a clear KA01H/R4656 plate became
KA01HR4651 and how two-line scooter plates (KA01K/H4341) were never read at all.

Strategy
  1. detect plate candidates (full snapshot + upscaled lower region for two-wheelers)
  2. per candidate: several preprocessing variants; two-line plates are split into
     lines (projection valley) and each line is read separately
  3. every read is repaired with Indian-plate rules + a 'KA' prior, then all variants vote
  4. if the plate detector found nothing / nothing valid: scan the lower snapshot with
     EasyOCR's own text detector and rebuild the plate from nearby text boxes
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from plate_utils import VALID_STATES, extract_valid_plate, sort_character_boxes_multiline

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PRIOR_STATE = "KA"   # dominant state in this deployment; used only to repair a damaged prefix


# --------------------------------------------------------------------------- text repair
def clean(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def repair(raw: str) -> Optional[str]:
    """Valid Indian plate from noisy text, or None."""
    s = clean(raw)
    if len(s) < 8:
        return None
    v = extract_valid_plate(s)
    if v:
        return v
    # damaged state prefix (e.g. 'K4', 'KR', 'XA'): retry with the prior state
    if s[:2] not in VALID_STATES and (s[0] == "K" or s[1] == "A"):
        v = extract_valid_plate(PRIOR_STATE + s[2:])
        if v:
            return v
    # one stray leading/trailing char (screw, border, 'IND')
    for t in (s[1:], s[:-1]):
        v = extract_valid_plate(t)
        if v:
            return v
    return None


def char_vote(valid: List[Tuple[str, float]]) -> Optional[str]:
    """Position-wise weighted vote across same-length valid reads (fixes single-digit slips)."""
    by_len: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for t, c in valid:
        by_len[len(t)].append((t, c))
    n, items = max(by_len.items(), key=lambda kv: (len(kv[1]), kv[0]))
    if len(items) < 3:
        return None
    out = []
    for i in range(n):
        w: Dict[str, float] = defaultdict(float)
        for t, c in items:
            w[t[i]] += 0.5 + c
        out.append(max(w, key=w.get))
    cand = "".join(out)
    return cand if extract_valid_plate(cand) == cand else None


# --------------------------------------------------------------------------- image helpers
def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()


def _norm_height(g: np.ndarray, target: int) -> np.ndarray:
    h, w = g.shape[:2]
    if h == target or h <= 0:
        return g
    f = target / float(h)
    interp = cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA
    return cv2.resize(g, (max(1, int(w * f)), target), interpolation=interp)


def _clahe(g: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)


def _sharpen(g: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    return cv2.addWeighted(g, 1.6, blur, -0.6, 0)


def split_two_lines(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Split a two-line plate at the horizontal-projection valley. Returns (top, bottom, found_valley)."""
    h, w = g.shape[:2]
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:          # make characters the minority (white-on-black)
        bw = 255 - bw
    m = max(2, int(0.06 * min(h, w)))
    inner = bw[m:h - m, m:w - m] if h > 2 * m + 4 and w > 2 * m + 4 else bw
    prof = inner.sum(axis=1).astype(np.float64) / 255.0
    k = max(3, len(prof) // 20)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    n = len(prof)
    lo, hi = int(n * 0.30), int(n * 0.70)
    idx = lo + int(np.argmin(prof[lo:hi]))
    top_peak, bot_peak = prof[:idx].max() if idx > 0 else 0, prof[idx:].max() if idx < n else 0
    found = prof[idx] < 0.55 * min(top_peak, bot_peak) if min(top_peak, bot_peak) > 0 else False
    cut = idx + m
    pad = max(1, int(h * 0.03))
    return g[: min(h, cut + pad)], g[max(0, cut - pad):], bool(found)


# --------------------------------------------------------------------------- OCR
def _read(reader, img: np.ndarray) -> Tuple[str, float]:
    res = reader.readtext(img, allowlist=ALLOW, detail=1, paragraph=False,
                          text_threshold=0.5, low_text=0.3)
    if not res:
        return "", 0.0
    res = sorted(res, key=lambda r: min(p[0] for p in r[0]))
    txt = clean("".join(r[1] for r in res))
    return txt, float(np.mean([r[2] for r in res]))


def read_plate_crop(reader, crop: np.ndarray) -> List[Tuple[str, float, str]]:
    """All text hypotheses (text, conf, tag) for one plate crop."""
    if crop is None or crop.size == 0:
        return []
    g = _to_gray(crop)
    h, w = g.shape[:2]
    two_line = (w / float(max(h, 1))) < 3.0
    out: List[Tuple[str, float, str]] = []
    if two_line:
        for name, fn in (("clahe", _clahe), ("sharp", _sharpen), ("raw", lambda x: x)):
            top, bot, _ = split_two_lines(fn(g))
            t_txt, t_c = _read(reader, _norm_height(top, 110))
            b_txt, b_c = _read(reader, _norm_height(bot, 110))
            if t_txt or b_txt:
                cs = [c for c, t in ((t_c, t_txt), (b_c, b_txt)) if t]
                out.append((t_txt + b_txt, float(np.mean(cs)), f"2line-{name}"))
        # whole-crop read as one more opinion (multi-line aware ordering)
        big = _norm_height(_clahe(g), 220)
        res = reader.readtext(big, allowlist=ALLOW, detail=1, paragraph=False,
                              text_threshold=0.5, low_text=0.3)
        if res:
            srt = sort_character_boxes_multiline(res)
            out.append((clean("".join(r[1] for r in srt)), float(np.mean([r[2] for r in srt])), "whole"))
    else:
        for name, fn in (("clahe", _clahe), ("sharp", _sharpen), ("raw", lambda x: x)):
            txt, c = _read(reader, _norm_height(fn(g), 128))
            if txt:
                out.append((txt, c, f"1line-{name}"))
    return out


def choose_best(cands: List[Tuple[str, float, str]]) -> Dict[str, Any]:
    """Vote across hypotheses. status: valid | partial | none."""
    valid: List[Tuple[str, float]] = []
    for t, c, _ in cands:
        v = repair(t)
        if v:
            valid.append((v, c))
    if valid:
        score: Dict[str, float] = defaultdict(float)
        confs: Dict[str, List[float]] = defaultdict(list)
        for v, c in valid:
            score[v] += 0.5 + c
            confs[v].append(c)
        best = max(score, key=score.get)
        voted = char_vote(valid)
        if voted and voted != best and score.get(voted, 0) == 0:
            best = voted
            confs[best] = [c for _, c in valid]
        return {"status": "valid", "plate": best, "conf": float(np.mean(confs[best])),
                "support": len(confs[best])}
    usable = [(clean(t), c) for t, c, _ in cands if len(clean(t)) >= 6]
    if usable:
        t, c = max(usable, key=lambda tc: (len(tc[0]) * tc[1]))
        return {"status": "partial", "plate": t, "conf": c, "support": 0}
    return {"status": "none", "plate": None, "conf": 0.0, "support": 0}


# --------------------------------------------------------------------------- detection
def detect_plate_boxes(model, img: np.ndarray, device: str, half: bool, two_wheeler: bool,
                       aspect: Tuple[float, float]) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Plate candidates in snapshot coordinates, best first."""
    H, W = img.shape[:2]
    found: List[Tuple[Tuple[int, int, int, int], float]] = []

    def run(im: np.ndarray, conf: float, ox: int, oy: int, s: float) -> None:
        sz = int(min(1280, ((max(im.shape[:2]) + 31) // 32) * 32))
        res = model.predict(im, conf=conf, iou=0.45, imgsz=sz, half=half, device=device, verbose=False)
        if not res or res[0].boxes is None:
            return
        for xyxy, c in zip(res[0].boxes.xyxy.cpu().numpy(), res[0].boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [int(v / s) for v in xyxy]
            x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy
            bw, bh = x2 - x1, y2 - y1
            if bw > 4 and bh > 4 and aspect[0] <= bw / float(bh) <= aspect[1]:
                found.append(((max(0, x1), max(0, y1), min(W, x2), min(H, y2)), float(c)))

    base = img
    s0 = 1.0
    if max(H, W) < 960:
        s0 = 960.0 / max(H, W)
        base = cv2.resize(img, None, fx=s0, fy=s0, interpolation=cv2.INTER_CUBIC)
    run(base, 0.08, 0, 0, s0)
    if two_wheeler:   # plates sit in the lower half of a bike+rider snapshot: zoom there
        y0 = int(H * 0.40)
        low = img[y0:]
        s1 = max(1.0, min(2.0, 1280.0 / max(low.shape[:2])))
        lowz = cv2.resize(low, None, fx=s1, fy=s1, interpolation=cv2.INTER_CUBIC) if s1 > 1.01 else low
        run(lowz, 0.05, 0, y0, s1)
    # merge duplicates (IoU) keep highest conf
    found.sort(key=lambda b: b[1] * ((b[0][2] - b[0][0]) * (b[0][3] - b[0][1])) ** 0.5, reverse=True)
    kept: List[Tuple[Tuple[int, int, int, int], float]] = []
    for b, c in found:
        ok = True
        for k, _ in kept:
            ix1, iy1, ix2, iy2 = max(b[0], k[0]), max(b[1], k[1]), min(b[2], k[2]), min(b[3], k[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if ua > 0 and inter / ua > 0.4:
                ok = False
                break
        if ok:
            kept.append((b, c))
    return kept


def scan_for_plate(reader, img: np.ndarray, lower_frac: float = 0.6):
    """Fallback when the plate detector fails: let EasyOCR find text, rebuild a plate from nearby boxes."""
    H, W = img.shape[:2]
    y0 = int(H * (1 - lower_frac))
    reg = img[y0:]
    res = reader.readtext(reg, allowlist=ALLOW, detail=1, paragraph=False, text_threshold=0.5, low_text=0.3)
    items = []
    for pts, txt, c in res:
        t = clean(txt)
        if len(t) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        items.append({"t": t, "c": float(c), "x1": min(xs), "x2": max(xs),
                      "y1": min(ys), "y2": max(ys), "cy": (min(ys) + max(ys)) / 2.0,
                      "h": max(ys) - min(ys), "cx": (min(xs) + max(xs)) / 2.0})
    best = None
    for seed in items:
        grp = [it for it in items
               if abs(it["cy"] - seed["cy"]) <= 2.2 * max(seed["h"], 1)
               and abs(it["cx"] - seed["cx"]) <= 3.5 * max(seed["x2"] - seed["x1"], seed["h"] * 3)]
        grp.sort(key=lambda it: it["cy"])
        rows: List[List[dict]] = []
        for it in grp:
            if rows and abs(it["cy"] - rows[-1][0]["cy"]) < 0.6 * max(it["h"], 1):
                rows[-1].append(it)
            else:
                rows.append([it])
        text = ""
        for r in rows:
            r.sort(key=lambda it: it["x1"])
            text += "".join(it["t"] for it in r)
        v = repair(text)
        if v:
            conf = float(np.mean([it["c"] for it in grp]))
            box = (int(min(i["x1"] for i in grp)), int(min(i["y1"] for i in grp)) + y0,
                   int(max(i["x2"] for i in grp)), int(max(i["y2"] for i in grp)) + y0)
            if best is None or conf * len(v) > best[1] * len(best[0]):
                best = (v, conf, box)
    return best
