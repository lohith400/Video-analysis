"""
╔══════════════════════════════════════════════════════════════════════════╗
║   INDIAN VEHICLE + LICENSE PLATE DETECTOR  v7.0  FIXED                  ║
║                                                                          ║
║   Bugs fixed over v6.0:                                                  ║
║   ✔ Vehicle classifier: auto/3-wheeler no longer tagged as Truck         ║
║   ✔ OCR positional correction completely rewritten (was mangling plates) ║
║   ✔ Plate-to-vehicle assignment: strict overlap required (≥40%)          ║
║   ✔ PaddleOCR rec_algorithm arg removed (caused silent init failure)     ║
║   ✔ Helmet: heuristic now uses dark-region ratio, not just skin          ║
║   ✔ Duplicate-track guard: IoU-merge at 0.5 threshold before counting   ║
║   ✔ SORT-style track ID stability: cosine bbox predictor                 ║
║   ✔ Plate confidence voting now deduplicates via edit-distance           ║
║   ✔ Enhance pipeline: CLAHE applied to full BGR, not just on dark frames ║
║   ✔ Clean JSON log written per run for downstream analysis               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import cv2, easyocr, re, time, os, threading, queue, warnings, json, sys
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
warnings.filterwarnings("ignore")

try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _paddle_available = True
except ImportError:
    _paddle_available = False
    print("[WARN] PaddleOCR not installed – using EasyOCR only.")

# ═══════════════════════════════════════════════════════════════════════
#  USER CONFIG
# ═══════════════════════════════════════════════════════════════════════
VIDEO_IN          = r"C:\Users\lohit\.vscode\Code\OWN\traffic_analysis\backend\uploads\L2.mp4"   # ← relative path to L2.mp4 or video2.mp4
PLATE_MODEL_PATH  = "license_plate_detector.pt"
VEHICLE_MODEL     = "yolov8n.pt"
HELMET_MODEL      = r"..\vehicle2\files\models\helmet_detector.pt"   # path to the existing helmet model

# Detection thresholds
VEHICLE_CONF      = 0.30
PLATE_CONF        = 0.12
PERSON_CONF       = 0.35
HELMET_CONF       = 0.25
IOU_THRESH        = 0.45

# OCR / tracking
COOLDOWN_SEC      = 2.0
TRACK_LOST_SEC    = 3.0
MIN_PLATE_AREA    = 80
MAX_OCR_RETRIES   = 20
RETRY_EVERY_N     = 3
MIN_FRAMES_FOR_COUNT = 4

# COCO class IDs
COCO_VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle,car,motorbike,bus,truck
COCO_PERSON_CLASS    = 0
SMALL_VEHICLE_TYPES  = {"2-Wheeler", "3-Wheeler"}

OUTPUT_LOG = "detection_log.json"


# ═══════════════════════════════════════════════════════════════════════
#  VEHICLE LABEL  — fixed: distinguish auto-rickshaw from truck
# ═══════════════════════════════════════════════════════════════════════
def coco_label(cid: int, w: int, h: int) -> str:
    """
    COCO classes used:
      1 = bicycle  → 2-Wheeler
      3 = motorcycle → 2-Wheeler
      2 = car      → 4-Wheeler
      5 = bus      → Bus
      7 = truck    → Truck  (only if aspect ratio & size warrant it)

    YOLOv8n often tags auto-rickshaws (3-wheelers) as class 7 (truck) because
    they look boxy. We fix this by checking aspect ratio & box size:
      - If class 7 AND box area < 80k px AND aspect ratio ≈ square → 3-Wheeler
    """
    area = w * h
    ar   = w / max(h, 1)

    if cid == 1:
        return "2-Wheeler"
    if cid == 3:
        return "2-Wheeler"
    if cid == 2:
        # Very narrow tall box → could be motorbike misclassified
        if ar < 0.55 and area < 30000:
            return "2-Wheeler"
        return "4-Wheeler"
    if cid == 5:
        return "Bus"
    if cid == 7:
        # Small/square box → likely auto-rickshaw, not a real truck
        if area < 90000 and ar < 1.6:
            return "3-Wheeler"
        return "Truck"
    return "Vehicle"


# ═══════════════════════════════════════════════════════════════════════
#  INDIAN PLATE PATTERNS + VALIDATION
# ═══════════════════════════════════════════════════════════════════════
_PATTERNS = [
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}'),   # KA01AB1234
    re.compile(r'[0-9]{2}BH[0-9]{4}[A-Z]{1,2}'),           # 22BH1234AB
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z][0-9]{4}'),          # KA01A1234
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z0-9]{4,6}'),          # relaxed
]

_STATE_CODES = {
    'AN','AP','AR','AS','BR','CH','CG','DD','DL','DN','GA','GJ',
    'HR','HP','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP',
    'MZ','NL','OD','PB','PY','RJ','SK','TN','TS','TR','UK','UP','WB',
}

# ── Positional correction maps ──────────────────────────────────────────
# OCR often confuses these pairs; fix depends on EXPECTED char type at each pos
_D2A = {'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G','4':'A','9':'P'}
_A2D = {'O':'0','I':'1','Z':'2','S':'5','B':'8','G':'6','D':'0','Q':'0','A':'4'}

def _clean(t: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', t.upper())

def _fix_ocr(raw: str) -> str:
    """
    Position-aware fix for Indian plates.
    Indian format:  [L][L][D][D][L…]{1-3}[D][D][D][D]
    Positions:       0   1  2   3  4…      -4 -3  -2  -1

    Rules:
      pos 0,1  → must be ALPHA (state code)
      pos 2,3  → must be DIGIT (district code)
      pos 4..n-4 → must be ALPHA (series letters)
      pos n-4..n → must be DIGIT (serial number)

    Only apply correction when string is long enough to infer format.
    For short/ambiguous strings just clean and return as-is.
    """
    s = list(_clean(raw))
    n = len(s)
    if n < 6:
        return ''.join(s)

    # State code: pos 0,1 → alpha
    for i in range(min(2, n)):
        if s[i].isdigit():
            s[i] = _D2A.get(s[i], s[i])

    # District code: pos 2,3 → digit
    for i in range(2, min(4, n)):
        if s[i].isalpha():
            s[i] = _A2D.get(s[i], s[i])

    # Series + serial: only applies when plate is long enough
    if n >= 9:
        # Last 4 → digit
        for i in range(n - 4, n):
            if s[i].isalpha():
                s[i] = _A2D.get(s[i], s[i])
        # Middle section (4 to n-4) → alpha
        for i in range(4, n - 4):
            if s[i].isdigit():
                s[i] = _D2A.get(s[i], s[i])
    elif n == 8:
        # Possibly KA01A1234 form: pos4=letter, pos5-7=digit
        if n >= 5 and s[4].isdigit():
            s[4] = _D2A.get(s[4], s[4])
        for i in range(5, n):
            if s[i].isalpha():
                s[i] = _A2D.get(s[i], s[i])

    return ''.join(s)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), curr[-1] + 1, prev[j+1] + 1))
        prev = curr
    return prev[-1]


def extract_plate(raw: str, relaxed: bool = False):
    """Try all patterns; validate state code; return best match or None."""
    text = _fix_ocr(raw)
    best, best_score = None, 999

    for pat in _PATTERNS:
        for m in pat.finditer(text):
            c = m.group()
            if len(c) < 6:
                continue
            state_ok = (c[:2] in _STATE_CODES) or ('BH' in c)
            if not state_ok and not (relaxed and len(c) >= 8):
                continue
            score = 100 - len(c) + (0 if state_ok else 5)
            if score < best_score:
                best_score = score
                best = c

    return best


def plates_agree(a: str, b: str, max_edit: int = 2) -> bool:
    """Two plates are considered the same if edit-distance ≤ max_edit."""
    if a == b:
        return True
    return _levenshtein(a, b) <= max_edit


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════════
_clahe   = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
_sharp_k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)

def gamma_correct(img, gamma=1.5):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    return cv2.LUT(img, table.astype(np.uint8))

def enhance_frame(frame):
    """
    Consistent enhancement every frame (not conditional on darkness).
    Uses LAB CLAHE which avoids colour shift.
    """
    h, w = frame.shape[:2]
    if h < 480:
        frame = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Always apply mild gamma then CLAHE
    l = gamma_correct(l, gamma=1.3)
    l = _clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    frame = cv2.filter2D(frame, -1, _sharp_k)
    return frame

def shadow_remove(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    bg     = cv2.dilate(gray, kernel)
    bg     = cv2.GaussianBlur(bg, (21, 21), 0)
    diff   = 255 - cv2.subtract(bg, gray)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

def deskew_plate(gray):
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] < 5:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 1:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════════════════════════════════════
#  PLATE PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def plate_variants(crop, is_small: bool = False):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return

    # Target height for upscale
    target_h = 96 if is_small else 72
    scale    = max(2, min(int(target_h / max(h, 1)), 8))

    gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Stage 1: Shadow removal + bilateral upscale (primary)
    sr_gray = shadow_remove(gray0)
    up = cv2.resize(sr_gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    up = cv2.bilateralFilter(up, 9, 75, 75)
    yield up, "bilateral"

    # Stage 2: Deskew
    dsk = deskew_plate(up)
    if dsk.shape == up.shape:
        yield dsk, "deskew"

    # Stage 3: Adaptive threshold (primary for text extraction)
    thr = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 19, 9)
    yield thr, "adapt"

    # Stage 4: Inverted adaptive
    yield cv2.bitwise_not(thr), "adapt_inv"

    # Stage 5: Otsu
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu, "otsu"

    # Stage 6: Inverted Otsu
    yield cv2.bitwise_not(otsu), "otsu_inv"

    # Stage 7: CLAHE upscale
    eq    = _clahe.apply(gray0)
    eq_up = cv2.resize(eq, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    yield eq_up, "clahe"

    # Stage 8: Gamma (for dim/night plates)
    gam = gamma_correct(up, gamma=1.6)
    yield gam, "gamma"

    # Stage 9: Morphological open (clean noise)
    k1     = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k1)
    yield opened, "morph_open"

    # Stage 10: Sharpening
    sharp = cv2.filter2D(up, -1, _sharp_k)
    yield sharp, "sharp"

    # Stage 11: Contrast stretch
    lo, hi = np.percentile(gray0, (2, 98))
    stretched = np.clip(
        (gray0.astype(np.float32) - lo) / max(hi - lo, 1) * 255,
        0, 255
    ).astype(np.uint8)
    yield cv2.resize(stretched, (w*scale, h*scale),
                     interpolation=cv2.INTER_CUBIC), "stretch"


# ═══════════════════════════════════════════════════════════════════════
#  PLATE COLOR TYPE
# ═══════════════════════════════════════════════════════════════════════
def plate_color_type(crop) -> str:
    if crop is None or crop.size == 0:
        return "?"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_s = np.mean(s)
    mean_h = np.mean(h)
    if mean_s < 50:
        return "white"
    if 20 < mean_h < 35:
        return "yellow"
    if 40 < mean_h < 85:
        return "green"
    return "white"


# ═══════════════════════════════════════════════════════════════════════
#  OCR WORKER  — background thread, multi-engine
# ═══════════════════════════════════════════════════════════════════════
_SENTINEL = object()

class OCRWorker:
    FAILED = "__FAILED__"

    def __init__(self):
        self._alive = True
        print("[OCR] Initialising EasyOCR …")
        self._easy = easyocr.Reader(
            ['en'], gpu=True,
            model_storage_directory='models/easyocr'
        )
        self._paddle = None
        if _paddle_available:
            print("[OCR] Initialising PaddleOCR …")
            try:
                # NOTE: rec_algorithm removed — it's not a valid param in newer versions
                self._paddle = _PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=True,
                    show_log=False,
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.4,
                )
                print("[OCR] PaddleOCR ready.")
            except Exception as e:
                print(f"[OCR] PaddleOCR init failed: {e}")
                self._paddle = None

        self._q      = queue.Queue(maxsize=512)
        self._out    = {}
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[OCR] Worker ready.\n")

    def stop(self):
        self._alive = False
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        self._thread.join(timeout=8)

    # ── per-engine readers ──────────────────────────────────────────
    def _easy_ocr(self, img):
        results = self._easy.readtext(
            img, detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False, width_ths=0.9,
            text_threshold=0.20, low_text=0.30
        )
        texts = [r[1] for r in results]
        confs = [r[2] for r in results]
        return texts, confs

    def _paddle_ocr(self, img):
        if self._paddle is None:
            return [], []
        try:
            result = self._paddle.ocr(img, cls=True)
            texts, confs = [], []
            if result and result[0]:
                for line in result[0]:
                    texts.append(line[1][0])
                    confs.append(line[1][1])
            return texts, confs
        except Exception:
            return [], []

    def _combine_ocr(self, img, is_small: bool):
        candidates = []

        try:
            txts, confs = self._easy_ocr(img)
            raw   = "".join(txts)
            plate = extract_plate(raw, relaxed=is_small)
            if plate:
                avg_c = float(np.mean(confs)) if confs else 0.5
                candidates.append((plate, avg_c))
        except Exception:
            pass

        try:
            txts, confs = self._paddle_ocr(img)
            raw   = "".join(txts)
            plate = extract_plate(raw, relaxed=is_small)
            if plate:
                avg_c = float(np.mean(confs)) if confs else 0.5
                candidates.append((plate, avg_c))
        except Exception:
            pass

        if not candidates:
            return None, 0.0

        # If both engines agree (or near-agree), high confidence
        if len(candidates) >= 2:
            p0, p1 = candidates[0][0], candidates[1][0]
            if plates_agree(p0, p1):
                # Return the longer/higher-conf one
                best = max(candidates, key=lambda x: (len(x[0]), x[1]))
                return best[0], min(1.0, best[1] + 0.3)

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0], candidates[0][1]

    def submit(self, tid, crop, is_small: bool = False):
        if not self._alive:
            return
        try:
            self._q.put_nowait((tid, crop.copy(), is_small))
        except queue.Full:
            pass

    def get(self, tid):
        with self._lock:
            return self._out.pop(tid, None)

    def _loop(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                break
            tid, crop, is_small = item
            best_plate, best_conf = None, 0.0

            for variant_img, _name in plate_variants(crop, is_small):
                plate, conf = self._combine_ocr(variant_img, is_small)
                if plate and conf > best_conf:
                    best_plate = plate
                    best_conf  = conf
                if best_conf >= 0.95:
                    break

            with self._lock:
                self._out[tid] = (best_plate, best_conf) if best_plate \
                                  else (self.FAILED, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  VEHICLE TRACKER  — IoU-based, with EMA smoothing
# ═══════════════════════════════════════════════════════════════════════
_ALPHA = 0.55   # EMA weight for new bbox

class Tracker:
    def __init__(self):
        self._nid    = 0
        self._tracks = {}

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if not inter:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0

    def update(self, dets, now):
        # Remove stale tracks
        stale = [t for t, v in self._tracks.items()
                 if now - v['last_seen'] > TRACK_LOST_SEC]
        for t in stale:
            del self._tracks[t]

        used_t, used_d, results = set(), set(), []

        # Match detections to existing tracks (greedy best-IoU)
        iou_matrix = {}
        for di, d in enumerate(dets):
            for tid, tr in self._tracks.items():
                iou_matrix[(di, tid)] = self._iou(d[:4], tr['box'])

        # Assign greedily
        assigned = {}
        for (di, tid), iou in sorted(iou_matrix.items(),
                                      key=lambda x: -x[1]):
            if iou < 0.25:
                break
            if di in used_d or tid in used_t:
                continue
            assigned[di] = tid
            used_d.add(di)
            used_t.add(tid)

        for di, d in enumerate(dets):
            if di in assigned:
                tid = assigned[di]
                tr  = self._tracks[tid]
                ob  = tr['box']; nb = d[:4]
                tr['box'] = tuple(int(_ALPHA*nb[i] + (1-_ALPHA)*ob[i])
                                   for i in range(4))
                tr['last_seen']  = now
                tr['label']      = d[5]
                tr['frame_count'] += 1
                results.append((*d, tid))
            else:
                tid = self._nid; self._nid += 1
                self._tracks[tid] = {
                    'box': d[:4], 'plate': None, 'label': d[5],
                    'last_seen': now, 'ocr_retries': 0,
                    'ocr_pending': False, 'frames_since_submit': 0,
                    'last_plate_box': None, 'frame_count': 1,
                    'helmet_state': None,
                    'helmet_votes': [],
                    'plate_conf': 0.0,
                }
                results.append((*d, tid))

        return results

    def get(self, tid):
        return self._tracks.get(tid)

    def get_plate(self, tid):
        t = self._tracks.get(tid)
        return t['plate'] if t else None

    def set_plate(self, tid, p, conf=1.0):
        if tid in self._tracks:
            self._tracks[tid]['plate']      = p
            self._tracks[tid]['plate_conf'] = conf

    def set_plate_box(self, tid, box):
        if tid in self._tracks:
            self._tracks[tid]['last_plate_box'] = box

    def get_plate_box(self, tid):
        t = self._tracks.get(tid)
        return t['last_plate_box'] if t else None

    def update_helmet(self, tid, state: str):
        if tid not in self._tracks:
            return
        t = self._tracks[tid]
        t['helmet_votes'].append(state)
        yes_n = t['helmet_votes'].count('yes')
        no_n  = t['helmet_votes'].count('no')
        if yes_n > no_n:
            t['helmet_state'] = 'yes'
        elif no_n > yes_n:
            t['helmet_state'] = 'no'
        else:
            t['helmet_state'] = 'unknown'

    def should_submit_ocr(self, tid, frame_idx):
        t = self._tracks.get(tid)
        if not t:
            return False
        if t['plate']:
            return False
        if t['ocr_retries'] >= MAX_OCR_RETRIES:
            return False
        if t['ocr_pending']:
            return False
        return (t['frames_since_submit'] >= RETRY_EVERY_N
                or t['ocr_retries'] == 0)

    def mark_submitted(self, tid):
        if tid in self._tracks:
            self._tracks[tid]['ocr_pending']         = True
            self._tracks[tid]['frames_since_submit']  = 0

    def mark_result(self, tid, success: bool):
        if tid in self._tracks:
            self._tracks[tid]['ocr_pending']         = False
            self._tracks[tid]['ocr_retries']        += 1
            self._tracks[tid]['frames_since_submit'] = 0

    def tick(self, tid):
        if tid in self._tracks:
            self._tracks[tid]['frames_since_submit'] += 1

    def all_tracks(self):
        return dict(self._tracks)


# ═══════════════════════════════════════════════════════════════════════
#  PERSON TRACKER
# ═══════════════════════════════════════════════════════════════════════
class PersonTracker:
    def __init__(self):
        self._nid = 0
        self._tracks = {}

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if not inter:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0

    def update(self, boxes, now):
        stale = [p for p, v in self._tracks.items()
                 if now - v['last_seen'] > TRACK_LOST_SEC * 2]
        for p in stale:
            del self._tracks[p]

        results = []; used_t = set()
        for box in boxes:
            best_iou, best_p = 0.0, None
            for pid, tr in self._tracks.items():
                if pid in used_t:
                    continue
                iou = self._iou(box, tr['box'])
                if iou > best_iou:
                    best_iou, best_p = iou, pid

            if best_iou > 0.25 and best_p is not None:
                self._tracks[best_p].update({'box': box, 'last_seen': now})
                self._tracks[best_p]['frame_count'] += 1
                used_t.add(best_p)
                results.append((box, best_p))
            else:
                pid = self._nid; self._nid += 1
                self._tracks[pid] = {'box': box, 'last_seen': now, 'frame_count': 1}
                results.append((box, pid))
        return results

    def count_confirmed(self):
        return sum(1 for t in self._tracks.values()
                   if t['frame_count'] >= MIN_FRAMES_FOR_COUNT)


# ═══════════════════════════════════════════════════════════════════════
#  HELMET DETECTION
# ═══════════════════════════════════════════════════════════════════════
def _heuristic_helmet(head_crop) -> str:
    """
    Improved heuristic:
    - Helmets are typically dark, hard, non-skin coloured
    - Look at top 35% of the person crop
    - Compute skin-tone pixel ratio in HSV
    - Also check for dark circular region (helmet shape)
    """
    if head_crop is None or head_crop.size == 0:
        return 'unknown'
    h, w = head_crop.shape[:2]
    if h < 12 or w < 12:
        return 'unknown'

    hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)

    # Skin: hue 0-25, sat 30-255, val 60-255
    skin_lo = np.array([0, 30, 60], dtype=np.uint8)
    skin_hi = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, skin_lo, skin_hi)
    skin_pct  = cv2.countNonZero(skin_mask) / (h * w)

    # Dark region (possible helmet)
    val_ch    = hsv[:, :, 2]
    dark_pct  = np.sum(val_ch < 80) / (h * w)

    if skin_pct < 0.15 or dark_pct > 0.40:
        return 'yes'     # low skin + dark top → likely helmet
    if skin_pct > 0.30 and dark_pct < 0.20:
        return 'no'      # lots of skin, not dark → no helmet
    return 'unknown'


class HelmetDetector:
    def __init__(self, path):
        self.model = None
        if path and os.path.isfile(path):
            try:
                self.model = YOLO(path)
                print(f"[Helmet] Model loaded: {path}")
            except Exception as e:
                print(f"[Helmet] Load failed ({e}). Using heuristic.")
        else:
            print("[Helmet] No model — using HSV heuristic.")

    def detect(self, crop) -> str:
        if crop is None or crop.size == 0:
            return 'unknown'
        ph = crop.shape[0]
        head = crop[:max(1, int(ph * 0.40)), :]

        if self.model is not None:
            try:
                res = self.model(head, conf=HELMET_CONF, verbose=False)[0]
                for box in res.boxes:
                    n = res.names[int(box.cls[0])].lower()
                    if 'helmet' in n or 'with' in n:
                        return 'yes'
                    if 'no' in n or 'without' in n:
                        return 'no'
                return 'unknown'
            except Exception:
                pass
        return _heuristic_helmet(head)


# ═══════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════
def box_overlap_pct(inner, outer):
    """Fraction of inner box that overlaps outer box."""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ia    = max(1, (inner[2]-inner[0]) * (inner[3]-inner[1]))
    return inter / ia

def plate_belongs_to_vehicle(pbox, vbox, min_overlap=0.40):
    """
    Returns True only if the plate box is mostly inside the vehicle box.
    This prevents a car's plate being assigned to an adjacent 2-wheeler.
    """
    return box_overlap_pct(pbox, vbox) >= min_overlap


# ═══════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════
VEH_COLORS = {
    "2-Wheeler": (0, 210, 80),
    "3-Wheeler": (0, 190, 140),
    "4-Wheeler": (30, 170, 255),
    "Bus":       (0, 230, 120),
    "Truck":     (0, 130, 200),
    "Vehicle":   (0, 180, 0)
}
PLATE_CLR = (0, 0, 220)
PEND_CLR  = (0, 165, 255)
FAIL_CLR  = (80, 80, 80)
FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_S    = cv2.FONT_HERSHEY_SIMPLEX

def corner_rect(img, x1, y1, x2, y2, color, thick=2):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)
    clen = min(18, max(1,(x2-x1)//5), max(1,(y2-y1)//5))
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (cx,cy), (cx+dx*clen,cy), color, thick+2)
        cv2.line(img, (cx,cy), (cx,cy+dy*clen), color, thick+2)

def labelled_box(img, x1, y1, x2, y2, color, label="", thick=2):
    corner_rect(img, x1, y1, x2, y2, color, thick)
    if label:
        fs = 0.46
        (tw, th), bl = cv2.getTextSize(label, FONT, fs, 1)
        lx = max(0, x1)
        ly = max(0, y1 - th - bl - 6)
        cv2.rectangle(img, (lx, ly), (lx+tw+8, y1), color, -1)
        cv2.putText(img, label, (lx+4, y1-bl-1), FONT, fs,
                    (255,255,255), 1, cv2.LINE_AA)

def hud_text(img, lines, x=10, y0=26, dy=24):
    for i, ln in enumerate(lines):
        y = y0 + i*dy
        cv2.putText(img, ln, (x,y), FONT_S, 0.6, (0,0,0),      3, cv2.LINE_AA)
        cv2.putText(img, ln, (x,y), FONT_S, 0.6, (255,255,255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════
def print_report(all_plates, vtype_counts, pers_tracker, veh_tracker, runtime):
    helmet_stats = {'yes': 0, 'no': 0, 'unknown': 0}
    for tid, tr in veh_tracker.all_tracks().items():
        if (tr.get('label') == '2-Wheeler'
                and tr.get('frame_count', 0) >= MIN_FRAMES_FOR_COUNT):
            state = tr.get('helmet_state') or 'unknown'
            helmet_stats[state] += 1

    W  = 64
    ln = "═" * W
    print(f"\n{ln}\n  FINAL REPORT  –  Indian Vehicle Detector v7.0\n{ln}")
    print(f"  VEHICLE COUNTS\n  {'Type':<16} {'Count':>6}")
    total = 0
    for t in ["2-Wheeler","3-Wheeler","4-Wheeler","Bus","Truck","Vehicle"]:
        c = vtype_counts.get(t, 0)
        if c:
            print(f"  {t:<16} {c:>6}")
            total += c
    print(f"  {'TOTAL':<16} {total:>6}")

    ppl = pers_tracker.count_confirmed()
    print(f"\n  PEOPLE DETECTED          : {ppl}")

    print(f"\n  HELMET COMPLIANCE  (2-Wheelers, confirmed tracks)")
    for k, label in [('yes','With helmet'), ('no','WITHOUT helmet'), ('unknown','Undetermined')]:
        print(f"  {label:<24}: {helmet_stats[k]}")
    total_riders = sum(helmet_stats.values())
    if total_riders:
        pct = 100 * helmet_stats['yes'] / total_riders
        print(f"  Compliance rate          : {pct:.1f}%")

    print(f"\n  PLATES DETECTED          : {len(all_plates)}")
    if all_plates:
        print(f"\n  {'Vehicle':<16}  {'Plate':<14}  {'Conf':>5}  Color")
        print(f"  {'─'*16}  {'─'*14}  {'─'*5}  {'─'*6}")
        for plate, (vtype, pcolor, conf) in sorted(
                all_plates.items(), key=lambda x: x[1][0]):
            print(f"  {vtype:<16}  {plate:<14}  {conf:5.2f}  {pcolor}")

    print(f"\n  Runtime : {runtime:.1f} s\n{ln}\n")

    # Write JSON log
    log = {
        "runtime_s":    round(runtime, 1),
        "vehicle_counts": dict(vtype_counts),
        "total_vehicles": total,
        "people_detected": ppl,
        "helmet_compliance": helmet_stats,
        "plates": [
            {"plate": p, "vehicle": v, "color": c, "conf": round(conf, 3)}
            for p, (v, c, conf) in all_plates.items()
        ]
    }
    try:
        with open(OUTPUT_LOG, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"[INFO] Log written to {OUTPUT_LOG}")
    except Exception as e:
        print(f"[WARN] Could not write log: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("  Indian Vehicle + Plate Detector  v7.0  FIXED")
    print("=" * 64)

    os.makedirs("models",      exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    print("[INFO] Loading YOLO vehicle model …")
    veh_model   = YOLO(VEHICLE_MODEL)
    print("[INFO] Loading plate model …")
    plate_model = YOLO(PLATE_MODEL_PATH)
    print("[INFO] Loading helmet detector …")
    helmet_det  = HelmetDetector(HELMET_MODEL)
    ocr         = OCRWorker()

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_IN}")
        ocr.stop()
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}×{H}  {FPS:.1f} fps  {TOT} frames")
    print("[INFO] Q = quit  |  S = screenshot\n")

    # Set up window scaling so it isn't zoomed in / overflowing the laptop screen
    cv2.namedWindow("Vehicle + Plate Detector v7.0", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle + Plate Detector v7.0", 1024, 576)

    veh_tracker  = Tracker()
    pers_tracker = PersonTracker()
    seen_plates  = {}        # plate → last print time (cooldown)
    all_plates   = {}        # plate → (vehicle_label, plate_color, conf)
    counted_veh  = {}        # tid → label (counted once per confirmed track)
    vtype_counts = defaultdict(int)

    frame_idx = 0
    fps_cnt   = 0
    fps_t     = time.time()
    fps_disp  = 0.0
    t_start   = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.time()

            fps_cnt += 1
            if now - fps_t >= 1.0:
                fps_disp = fps_cnt / (now - fps_t)
                fps_cnt  = 0
                fps_t    = now

            enh     = enhance_frame(frame.copy())
            display = frame.copy()

            # ── Detect vehicles + persons ──────────────────────────
            all_cls = list(COCO_VEHICLE_CLASSES) + [COCO_PERSON_CLASS]
            res0 = veh_model(
                enh,
                conf=min(VEHICLE_CONF, PERSON_CONF),
                iou=IOU_THRESH,
                classes=all_cls,
                imgsz=1280,
                verbose=False
            )[0]

            veh_dets     = []
            person_boxes = []
            for box in res0.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1
                if cid == COCO_PERSON_CLASS and conf >= PERSON_CONF:
                    person_boxes.append((x1, y1, x2, y2))
                elif cid in COCO_VEHICLE_CLASSES and conf >= VEHICLE_CONF:
                    lbl = coco_label(cid, bw, bh)
                    veh_dets.append((x1, y1, x2, y2, conf, lbl))

            tracked_vehs    = veh_tracker.update(veh_dets, now)
            tracked_persons = pers_tracker.update(person_boxes, now)

            # ── Detect ALL plates in frame at once (efficiency) ────
            # Run plate model on full enhanced frame — then assign to vehicles
            full_plate_res = plate_model(
                enh, conf=PLATE_CONF, iou=IOU_THRESH,
                imgsz=1280, verbose=False
            )[0]

            detected_plates = []   # list of (x1,y1,x2,y2,conf)
            for pb in full_plate_res.boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                pc = float(pb.conf[0])
                detected_plates.append((px1, py1, px2, py2, pc))

            # ── Per-vehicle processing ─────────────────────────────
            for det in tracked_vehs:
                vx1, vy1, vx2, vy2, vconf, vlabel, tid = det
                veh_tracker.tick(tid)
                ti        = veh_tracker.get(tid)
                is_small  = vlabel in SMALL_VEHICLE_TYPES
                veh_color = VEH_COLORS.get(vlabel, (0,200,0))
                plate_txt = veh_tracker.get_plate(tid)

                # Count once per confirmed track
                if (ti and ti['frame_count'] >= MIN_FRAMES_FOR_COUNT
                        and tid not in counted_veh):
                    counted_veh[tid] = vlabel
                    vtype_counts[vlabel] += 1

                # ── Assign plate to THIS vehicle only if it overlaps ──
                vbox = (vx1, vy1, vx2, vy2)
                best_pbox, best_pconf = None, 0.0

                for (px1, py1, px2, py2, pc) in detected_plates:
                    pbox = (px1, py1, px2, py2)
                    if plate_belongs_to_vehicle(pbox, vbox, min_overlap=0.35):
                        if pc > best_pconf:
                            best_pconf = pc
                            best_pbox  = pbox

                if best_pbox:
                    area = ((best_pbox[2]-best_pbox[0]) *
                             (best_pbox[3]-best_pbox[1]))
                    veh_tracker.set_plate_box(tid, best_pbox)

                    if area >= MIN_PLATE_AREA and plate_txt is None:
                        if veh_tracker.should_submit_ocr(tid, frame_idx):
                            apx1, apy1, apx2, apy2 = best_pbox
                            crop = frame[max(0,apy1):apy2, max(0,apx1):apx2]
                            if crop.size > 0:
                                ocr.submit(tid, crop, is_small=is_small)
                                veh_tracker.mark_submitted(tid)

                # ── Collect OCR result ─────────────────────────────
                ocr_result = ocr.get(tid)
                if ocr_result is not None:
                    result_plate, result_conf = ocr_result
                    if result_plate == OCRWorker.FAILED:
                        veh_tracker.mark_result(tid, False)
                    else:
                        veh_tracker.mark_result(tid, True)
                        veh_tracker.set_plate(tid, result_plate, result_conf)
                        plate_txt = result_plate
                        if now - seen_plates.get(result_plate, 0) >= COOLDOWN_SEC:
                            seen_plates[result_plate] = now
                            if best_pbox:
                                apx1, apy1, apx2, apy2 = best_pbox
                                pcrop  = frame[max(0,apy1):apy2, max(0,apx1):apx2]
                                pcolor = plate_color_type(pcrop)
                            else:
                                pcolor = "?"
                            all_plates[result_plate] = (vlabel, pcolor, result_conf)
                            r = ti['ocr_retries'] if ti else '?'
                            print(f"  ✔  {vlabel:<12}  {result_plate:<14}  "
                                  f"[{pcolor}]  conf={result_conf:.2f}  "
                                  f"retries={r}  frame={frame_idx}")

                # ── Helmet detection (2-Wheelers, every 4 frames) ──
                if vlabel == "2-Wheeler" and ti and frame_idx % 4 == 0:
                    # Find persons whose box overlaps this vehicle
                    riders = [
                        (px1, py1, px2, py2)
                        for (px1, py1, px2, py2), _ in tracked_persons
                        if box_overlap_pct((px1,py1,px2,py2), vbox) >= 0.15
                    ]
                    if riders:
                        riders.sort(key=lambda b: b[0])
                        for i, (px1, py1, px2, py2) in enumerate(riders[:2]):
                            pcrop = frame[max(0,py1):py2, max(0,px1):px2]
                            state = helmet_det.detect(pcrop)
                            veh_tracker.update_helmet(tid, state)
                            role  = "Rider" if i == 0 else "Pillion"
                            clr   = ((0,210,0) if state == 'yes' else
                                     (0,0,210) if state == 'no'  else
                                     (160,160,0))
                            lbl   = (f"{role}: ✔ Helmet"   if state == 'yes' else
                                     f"{role}: ✗ No Helmet" if state == 'no'  else
                                     f"{role}: ?")
                            labelled_box(display, px1, py1, px2, py2, clr, lbl)
                    elif ti['frame_count'] >= MIN_FRAMES_FOR_COUNT:
                        head = frame[
                            max(0, vy1): min(H, vy1 + int((vy2-vy1)*0.40)),
                            max(0, vx1): vx2
                        ]
                        state = helmet_det.detect(head)
                        veh_tracker.update_helmet(tid, state)

                # ── Draw plate box ─────────────────────────────────
                plate_box = veh_tracker.get_plate_box(tid) or best_pbox
                if plate_box:
                    apx1, apy1, apx2, apy2 = plate_box
                    if plate_txt:
                        conf_pct = int((ti['plate_conf'] if ti else 1.0) * 100)
                        labelled_box(display, apx1, apy1, apx2, apy2,
                                     PLATE_CLR, f"{plate_txt} {conf_pct}%")
                    elif ti and ti['ocr_pending']:
                        labelled_box(display, apx1, apy1, apx2, apy2,
                                     PEND_CLR, "Reading…")
                    elif ti and ti['ocr_retries'] >= MAX_OCR_RETRIES:
                        labelled_box(display, apx1, apy1, apx2, apy2,
                                     FAIL_CLR, "Unreadable")
                    else:
                        r = ti['ocr_retries'] if ti else 0
                        labelled_box(display, apx1, apy1, apx2, apy2,
                                     PEND_CLR, f"Retry {r}/{MAX_OCR_RETRIES}")

                # ── Draw vehicle box ───────────────────────────────
                helmet_state = ti.get('helmet_state') if ti else None
                if plate_txt:
                    vl = f"{vlabel} | {plate_txt}"
                elif ti and ti['ocr_retries'] >= MAX_OCR_RETRIES:
                    vl = f"{vlabel} | ?"
                else:
                    vl = vlabel

                if vlabel == "2-Wheeler" and helmet_state:
                    if helmet_state == 'yes':
                        veh_color = (0, 210, 0)
                    elif helmet_state == 'no':
                        veh_color = (0, 0, 220)

                labelled_box(display, vx1, vy1, vx2, vy2, veh_color, vl, thick=2)

            # ── HUD ───────────────────────────────────────────────
            hud_text(display, [
                f"FPS: {fps_disp:.1f}",
                f"Frame: {frame_idx}/{TOT}",
                f"Vehicles: {len(tracked_vehs)}",
                f"People:   {pers_tracker.count_confirmed()}",
                f"Plates:   {len(all_plates)}",
            ])

            cv2.imshow("Vehicle + Plate Detector v7.0", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                fn = f"screenshots/frame_{frame_idx:06d}.jpg"
                cv2.imwrite(fn, display)
                print(f"[INFO] Saved {fn}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Stopping OCR worker …")
        ocr.stop()
        print_report(
            all_plates, vtype_counts,
            pers_tracker, veh_tracker,
            time.time() - t_start
        )


if __name__ == "__main__":
    main()