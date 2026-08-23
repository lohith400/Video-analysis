"""
╔══════════════════════════════════════════════════════════════════════╗
║   INDIAN VEHICLE + LICENSE PLATE DETECTOR  v4.0  PRODUCTION          ║
║   ✔ Helmet Detection (rider + pillion)                                ║
║   ✔ Vehicle Type Counts in Final Report                               ║
║   ✔ Total People Counter                                              ║
║   ✔ Fixed 2-Wheeler & 3-Wheeler Plate OCR                             ║
║   ✔ Deskew + Multi-scale Plate Preprocessing                          ║
╚══════════════════════════════════════════════════════════════════════╝

MODELS NEEDED
─────────────────────────────────────────────────────────────────────
  • yolov8n.pt               — COCO vehicles + persons (auto-downloaded)
  • license_plate_detector.pt — fine-tuned plate detector
  • helmet_detector.pt        — (OPTIONAL) detects helmet / no-helmet
                                If missing → uses HSV head-colour heuristic

USAGE
  python vehicle_detector_v4.py
  Press  Q  to quit  |  S  to screenshot
"""

# ══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════
import cv2
import easyocr
import re
import time
import os
import threading
import queue
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════
#  ┌─────────────────────┐
#  │   USER CONFIG       │  ← edit these
#  └─────────────────────┘
# ══════════════════════════════════════════════════════════════════════
VIDEO_IN         = "/home/lohit/realcode/ML/Code/projectsss/vehicle/videos/L2.mp4"
PLATE_MODEL_PATH = "license_plate_detector.pt"
VEHICLE_MODEL    = "yolov8n.pt"
HELMET_MODEL     = "helmet_detector.pt"   # set to None to force heuristic

# ── Detection thresholds ──
VEHICLE_CONF     = 0.35
PLATE_CONF       = 0.15   # lower = catches small 2/3-wheeler plates too
PERSON_CONF      = 0.40
HELMET_CONF      = 0.45
IOU_THRESH       = 0.45

# ── Tracking / OCR ──
COOLDOWN_SEC     = 4.0
TRACK_LOST_SEC   = 3.0
MIN_PLATE_AREA   = 120    # smaller so 2/3-wheeler plates aren't dropped
MAX_OCR_RETRIES  = 10
RETRY_EVERY_N    = 4

# ── Per-vehicle-type OCR tuning ──
# 4-wheelers have larger plates; 2/3-wheelers need extra pre-processing
SMALL_VEHICLE_TYPES = {"2-Wheeler", "3-Wheeler"}

# ── Reporting ──
# Only count a vehicle in the final report if its track survived this many frames
# (avoids counting ghost/false-positive detections)
MIN_FRAMES_FOR_COUNT = 8

# ── Enhancement flags ──
ENHANCE_CONTRAST = True
SHARPEN          = True
UPSCALE_SMALL    = True

COCO_VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle, car, motorbike, bus, truck
COCO_PERSON_CLASS    = 0


# ══════════════════════════════════════════════════════════════════════
#  VEHICLE LABEL
# ══════════════════════════════════════════════════════════════════════
def coco_label(class_id: int, w: int, h: int) -> str:
    mapping = {1: "2-Wheeler", 3: "2-Wheeler", 2: "4-Wheeler",
               5: "Bus",       7: "Truck"}
    label = mapping.get(class_id, "Vehicle")
    # motorbike with wide aspect → likely auto-rickshaw
    if class_id == 3 and w > 0 and h > 0:
        if (w / h) > 1.2 and w > 80:
            label = "3-Wheeler"
    return label


# ══════════════════════════════════════════════════════════════════════
#  INDIAN PLATE PATTERNS  (covers all current series)
# ══════════════════════════════════════════════════════════════════════
_PATTERNS = [
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}'),   # standard
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}'),      # old series
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z][0-9]{4}'),         # single-letter mid
    re.compile(r'[0-9]{2}BH[0-9]{4}[A-Z]{1,2}'),          # BH (Bharat) series
    # relaxed fallback — accept if ≥ 8 chars and starts with 2 letters+2 digits
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z0-9]{4,6}'),
]

_STATE_CODES = {
    'AN','AP','AR','AS','BR','CH','CG','DD','DL','DN','GA','GJ','HR',
    'HP','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL',
    'OD','PB','PY','RJ','SK','TN','TS','TR','UK','UP','WB',
}

def _clean(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def _fix_ocr(text: str) -> str:
    """Correct common OCR mis-reads in Indian plates."""
    t = list(text)
    digit_to_alpha = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
    alpha_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}
    # First 2 chars must be letters (state code)
    for i in range(min(2, len(t))):
        if t[i].isdigit():
            t[i] = digit_to_alpha.get(t[i], t[i])
    # Chars 2-4 must be digits (district code)
    for i in range(2, min(4, len(t))):
        if t[i].isalpha():
            t[i] = alpha_to_digit.get(t[i], t[i])
    return ''.join(t)

def extract_plate(raw: str, relaxed: bool = False):
    """
    Try all patterns; validate state code.
    relaxed=True accepts partial matches when strict validation fails
    (used for 2/3-wheelers where plate crop may be noisy).
    """
    text = _fix_ocr(_clean(raw))
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            c = m.group()
            if len(c) >= 6:
                # strict: state code must match
                if c[:2] in _STATE_CODES or 'BH' in c:
                    return c
                # relaxed: still return if at least 8 chars and well-formed
                if relaxed and len(c) >= 8:
                    return c
    return None


# ══════════════════════════════════════════════════════════════════════
#  IMAGE ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════
_clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_sharp_k   = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)

def enhance_frame(frame):
    h, w = frame.shape[:2]
    if UPSCALE_SMALL and h < 480:
        frame = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    if ENHANCE_CONTRAST:
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        frame = cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    if SHARPEN:
        frame = cv2.filter2D(frame, -1, _sharp_k)
    return frame

def deskew_plate(gray):
    """Straighten a tilted plate crop using moments."""
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] < 5:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 1:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════════
#  PLATE PREPROCESSING PIPELINE  (7 stages)
# ══════════════════════════════════════════════════════════════════════
def plate_variants(crop, is_small_vehicle: bool = False):
    """
    Yield progressively processed versions of a plate crop for OCR.
    is_small_vehicle → extra upscaling + deskew for 2/3-wheelers.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return

    # More aggressive upscaling for small vehicle plates
    base_target = 80 if is_small_vehicle else 60
    scale = max(3 if is_small_vehicle else 2, int(base_target / max(h, 1)))
    scale = min(scale, 8)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Stage 0 — bilateral-filtered upscale
    up   = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    blur = cv2.bilateralFilter(up, 11, 80, 80)
    yield blur

    # Stage 1 — deskew + bilateral (critical for 2/3-wheelers)
    dsk = deskew_plate(blur)
    yield dsk

    # Stage 2 — adaptive threshold (bright ambient)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 21, 10)
    yield thr

    # Stage 3 — OTSU binary
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu

    # Stage 4 — inverted OTSU (dark-on-light plates)
    yield cv2.bitwise_not(otsu)

    # Stage 5 — CLAHE equalised
    eq  = _clahe.apply(gray)
    eq_up = cv2.resize(eq, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    yield eq_up

    # Stage 6 — morphological opening to remove noise specks
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    yield opened


# ══════════════════════════════════════════════════════════════════════
#  OCR WORKER  (background thread)
# ══════════════════════════════════════════════════════════════════════
class OCRWorker:
    FAILED = "__FAILED__"

    def __init__(self):
        print("[OCR] Initialising EasyOCR …")
        self.reader = easyocr.Reader(['en'], gpu=True)
        self._q     = queue.Queue(maxsize=256)
        self._out   = {}
        self._lock  = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        print("[OCR] Ready.\n")

    def submit(self, tid, crop, is_small: bool = False):
        try:
            self._q.put_nowait((tid, crop.copy(), is_small))
        except queue.Full:
            pass

    def get(self, tid):
        with self._lock:
            return self._out.pop(tid, None)

    def _run_ocr(self, img):
        return self.reader.readtext(
            img, detail=0,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False, width_ths=0.8, text_threshold=0.35
        )

    def _loop(self):
        while True:
            tid, crop, is_small = self._q.get()
            best = None
            for variant in plate_variants(crop, is_small_vehicle=is_small):
                try:
                    raw_list = self._run_ocr(variant)
                    if raw_list:
                        raw = "".join(raw_list)
                        plate = extract_plate(raw, relaxed=is_small)
                        if plate:
                            best = plate
                            break
                except Exception:
                    continue
            with self._lock:
                self._out[tid] = best if best else self.FAILED


# ══════════════════════════════════════════════════════════════════════
#  IoU TRACKER
# ══════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self._nid    = 0
        self._tracks = {}

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if not inter:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0

    def update(self, dets, now):
        # Expire stale tracks
        stale = [t for t, v in self._tracks.items()
                 if now - v['last_seen'] > TRACK_LOST_SEC]
        for t in stale:
            del self._tracks[t]

        used_t, used_d, results = set(), set(), []

        for di, d in enumerate(dets):
            best_iou, best_t = 0.0, None
            for tid, tr in self._tracks.items():
                if tid in used_t: continue
                iou = self._iou(d[:4], tr['box'])
                if iou > best_iou:
                    best_iou, best_t = iou, tid
            if best_iou > 0.3 and best_t is not None:
                tr = self._tracks[best_t]
                tr.update({'box': d[:4], 'last_seen': now, 'label': d[5]})
                tr['frame_count'] += 1
                used_t.add(best_t); used_d.add(di)
                results.append((*d, best_t))

        for di, d in enumerate(dets):
            if di in used_d: continue
            tid = self._nid; self._nid += 1
            self._tracks[tid] = {
                'box': d[:4], 'plate': None, 'label': d[5],
                'last_seen': now, 'ocr_retries': 0,
                'ocr_pending': False, 'frames_since_submit': 0,
                'last_plate_box': None, 'frame_count': 1,
                # helmet tracking per rider slot
                'helmet_states': [],   # list of 'yes'/'no'/'unknown' per rider
            }
            results.append((*d, tid))
        return results

    # ── accessors ──
    def get(self, tid):            return self._tracks.get(tid)
    def get_plate(self, tid):
        t = self._tracks.get(tid); return t['plate'] if t else None

    def set_plate(self, tid, p):
        if tid in self._tracks: self._tracks[tid]['plate'] = p

    def set_plate_box(self, tid, box):
        if tid in self._tracks: self._tracks[tid]['last_plate_box'] = box

    def get_plate_box(self, tid):
        t = self._tracks.get(tid)
        return t['last_plate_box'] if t else None

    def set_helmet_states(self, tid, states):
        if tid in self._tracks: self._tracks[tid]['helmet_states'] = states

    # ── OCR retry logic ──
    def should_submit_ocr(self, tid, frame_idx):
        t = self._tracks.get(tid)
        if not t: return False
        if t['plate']: return False
        if t['ocr_retries'] >= MAX_OCR_RETRIES: return False
        if t['ocr_pending']: return False
        return (t['frames_since_submit'] >= RETRY_EVERY_N
                or t['ocr_retries'] == 0)

    def mark_submitted(self, tid):
        if tid in self._tracks:
            self._tracks[tid]['ocr_pending'] = True
            self._tracks[tid]['frames_since_submit'] = 0

    def mark_result(self, tid, success: bool):
        if tid in self._tracks:
            self._tracks[tid]['ocr_pending'] = False
            self._tracks[tid]['ocr_retries'] += 1
            self._tracks[tid]['frames_since_submit'] = 0

    def tick(self, tid):
        if tid in self._tracks:
            self._tracks[tid]['frames_since_submit'] += 1

    # ── report helpers ──
    def all_confirmed_tracks(self):
        """Yield (tid, track_info) for tracks with enough frames."""
        for tid, t in self._tracks.items():
            if t['frame_count'] >= MIN_FRAMES_FOR_COUNT:
                yield tid, t


# ══════════════════════════════════════════════════════════════════════
#  PERSON TRACKER  (simple IoU tracker for persons)
# ══════════════════════════════════════════════════════════════════════
class PersonTracker:
    def __init__(self):
        self._nid    = 0
        self._tracks = {}  # pid → {box, last_seen, frame_count}
        self.all_ids = set()

    @staticmethod
    def _iou(a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if not inter: return 0.0
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua else 0.0

    def update(self, person_boxes, now):
        stale = [p for p, v in self._tracks.items()
                 if now - v['last_seen'] > TRACK_LOST_SEC * 2]
        for p in stale: del self._tracks[p]

        results = []
        used_t = set()
        for box in person_boxes:
            best_iou, best_p = 0.0, None
            for pid, tr in self._tracks.items():
                if pid in used_t: continue
                iou = self._iou(box, tr['box'])
                if iou > best_iou:
                    best_iou, best_p = iou, pid
            if best_iou > 0.3 and best_p is not None:
                self._tracks[best_p]['box'] = box
                self._tracks[best_p]['last_seen'] = now
                self._tracks[best_p]['frame_count'] += 1
                used_t.add(best_p)
                results.append((box, best_p))
            else:
                pid = self._nid; self._nid += 1
                self.all_ids.add(pid)
                self._tracks[pid] = {'box': box, 'last_seen': now,
                                     'frame_count': 1}
                results.append((box, pid))
        return results

    def count_confirmed(self):
        return sum(1 for t in self._tracks.values()
                   if t['frame_count'] >= MIN_FRAMES_FOR_COUNT)


# ══════════════════════════════════════════════════════════════════════
#  HELMET DETECTION
# ══════════════════════════════════════════════════════════════════════
def _heuristic_helmet(head_crop) -> str:
    """
    HSV-based heuristic.
    Helmets typically have a large region of uniform, non-skin colour.
    Returns 'yes', 'no', or 'unknown'.
    """
    if head_crop is None or head_crop.size == 0:
        return 'unknown'
    h, w = head_crop.shape[:2]
    if h < 10 or w < 10:
        return 'unknown'

    hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)

    # Skin-colour range in HSV
    skin_lo = np.array([0,  20, 70],  dtype=np.uint8)
    skin_hi = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, skin_lo, skin_hi)

    total = h * w
    skin_pct = cv2.countNonZero(skin_mask) / total

    # If very little skin → probably wearing a helmet
    if skin_pct < 0.15:
        return 'yes'
    # Lots of skin → no helmet
    elif skin_pct > 0.40:
        return 'no'
    return 'unknown'


class HelmetDetector:
    """
    Uses a dedicated YOLO helmet model when available,
    falls back to HSV heuristic otherwise.
    """
    def __init__(self, model_path):
        self.model = None
        if model_path and os.path.isfile(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"[Helmet] Model loaded: {model_path}")
            except Exception as e:
                print(f"[Helmet] Could not load model ({e}). Using heuristic.")
        else:
            print("[Helmet] No model found — using HSV heuristic.")

    def detect(self, person_crop) -> str:
        """Returns 'yes', 'no', or 'unknown'."""
        if person_crop is None or person_crop.size == 0:
            return 'unknown'
        ph = person_crop.shape[0]
        # Head region = top 35 % of person bounding box
        head_crop = person_crop[:max(1, int(ph * 0.35)), :]

        if self.model is not None:
            try:
                res = self.model(head_crop, conf=HELMET_CONF,
                                  verbose=False)[0]
                for box in res.boxes:
                    cls_name = res.names[int(box.cls[0])].lower()
                    if 'helmet' in cls_name or 'with' in cls_name:
                        return 'yes'
                    if 'no' in cls_name or 'without' in cls_name:
                        return 'no'
                return 'unknown'
            except Exception:
                pass
        return _heuristic_helmet(head_crop)


# ══════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════
VEH_COLOR_MAP = {
    "2-Wheeler": (0, 210,  80),
    "3-Wheeler": (0, 190, 140),
    "4-Wheeler": (0, 170,  50),
    "Bus":       (0, 230, 120),
    "Truck":     (0, 200,  60),
    "Vehicle":   (0, 180,   0),
}
PLATE_COLOR   = (0,   0, 230)
PEND_COLOR    = (0, 165, 255)
FAIL_COLOR    = (80, 80,  80)
HELMET_YES    = (0, 210,   0)
HELMET_NO     = (0,   0, 210)
HELMET_UNK    = (160, 160,  0)
FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_S        = cv2.FONT_HERSHEY_SIMPLEX

def corner_rect(img, x1, y1, x2, y2, color, thick=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    clen = min(16, max(1, (x2-x1)//5), max(1, (y2-y1)//5))
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (cx, cy), (cx+dx*clen, cy), color, thick+2)
        cv2.line(img, (cx, cy), (cx, cy+dy*clen), color, thick+2)

def labelled_box(img, x1, y1, x2, y2, color, label="", thick=2):
    corner_rect(img, x1, y1, x2, y2, color, thick)
    if label:
        fs = 0.46
        (tw, th), bl = cv2.getTextSize(label, FONT, fs, 1)
        lx = max(0, x1)
        ly = max(0, y1 - th - bl - 6)
        cv2.rectangle(img, (lx, ly), (lx+tw+8, y1), color, -1)
        cv2.putText(img, label, (lx+4, y1-bl-1),
                    FONT, fs, (255,255,255), 1, cv2.LINE_AA)

def hud_text(img, lines, x=10, y_start=26, dy=24):
    for i, line in enumerate(lines):
        y = y_start + i * dy
        cv2.putText(img, line, (x, y), FONT_S, 0.6, (0,0,0),   3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), FONT_S, 0.6, (255,255,255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════
#  OVERLAP HELPERS
# ══════════════════════════════════════════════════════════════════════
def box_overlap_pct(inner, outer):
    """What fraction of inner is inside outer? (0-1)"""
    ix1=max(inner[0],outer[0]); iy1=max(inner[1],outer[1])
    ix2=min(inner[2],outer[2]); iy2=min(inner[3],outer[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    inner_area=(inner[2]-inner[0])*(inner[3]-inner[1])
    return inter/inner_area if inner_area else 0.0


# ══════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ══════════════════════════════════════════════════════════════════════
def print_report(all_plates, vehicle_type_counts, person_tracker,
                 helmet_stats, runtime_sec):
    W = 62
    line = "═" * W
    print(f"\n{line}")
    print(f"  FINAL REPORT")
    print(line)

    # ── Vehicle counts ──
    print(f"  VEHICLE COUNTS")
    print(f"  {'Type':<16} {'Unique Tracked':>14}")
    print(f"  {'─'*16} {'─'*14}")
    total_veh = 0
    for vtype in ["2-Wheeler","3-Wheeler","4-Wheeler","Bus","Truck","Vehicle"]:
        cnt = vehicle_type_counts.get(vtype, 0)
        if cnt:
            print(f"  {vtype:<16} {cnt:>14}")
            total_veh += cnt
    print(f"  {'TOTAL':<16} {total_veh:>14}")

    # ── People ──
    ppl = person_tracker.count_confirmed()
    print(f"\n  PEOPLE DETECTED        : {ppl}")

    # ── Helmets ──
    print(f"\n  HELMET COMPLIANCE  (2-Wheelers)")
    print(f"  Riders with helmet    : {helmet_stats['yes']}")
    print(f"  Riders WITHOUT helmet : {helmet_stats['no']}")
    print(f"  Undetermined          : {helmet_stats['unknown']}")
    total_riders = sum(helmet_stats.values())
    if total_riders:
        pct = 100 * helmet_stats['yes'] / total_riders
        print(f"  Compliance rate       : {pct:.1f}%")

    # ── Plates ──
    print(f"\n  PLATES DETECTED        : {len(all_plates)}")
    print(f"\n  {'Vehicle':<16}  {'Plate':<14}")
    print(f"  {'─'*16}  {'─'*14}")
    for plate, vtype in sorted(all_plates.items(), key=lambda x: x[1]):
        print(f"  {vtype:<16}  {plate:<14}")

    print(f"\n  Runtime : {runtime_sec:.1f} s")
    print(line + "\n")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  Indian Vehicle + Plate Detector  v4.0")
    print("=" * 62)

    # ── Load models ──
    print("[INFO] Loading vehicle/person model …")
    veh_model   = YOLO(VEHICLE_MODEL)

    print("[INFO] Loading plate model …")
    plate_model = YOLO(PLATE_MODEL_PATH)

    print("[INFO] Initialising helmet detector …")
    helmet_det  = HelmetDetector(HELMET_MODEL)

    ocr         = OCRWorker()

    # ── Open video ──
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_IN}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}×{H}  {FPS:.1f} fps  {TOT} frames")
    print("[INFO] Q = quit  |  S = screenshot\n")

    # ── State ──
    veh_tracker  = Tracker()
    pers_tracker = PersonTracker()
    seen_plates  = {}          # plate → last_print_time
    all_plates   = {}          # plate → vehicle_label
    # persistent vehicle-type counter keyed by track-id (counted once)
    counted_veh  = {}          # tid → label  (so we only count each track once)
    vehicle_type_counts = defaultdict(int)
    # lifetime helmet stats across all 2-Wheeler riders
    helmet_stats = {'yes': 0, 'no': 0, 'unknown': 0}

    frame_idx = 0
    fps_cnt = 0; fps_t = time.time(); fps_disp = 0.0
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        now = time.time()

        fps_cnt += 1
        if now - fps_t >= 1.0:
            fps_disp = fps_cnt / (now - fps_t)
            fps_cnt  = 0; fps_t = now

        enh     = enhance_frame(frame.copy())
        display = frame.copy()

        # ════════════════════════════════════════════════
        # STAGE 1 — Detect vehicles (+ persons together)
        # ════════════════════════════════════════════════
        all_classes = list(COCO_VEHICLE_CLASSES) + [COCO_PERSON_CLASS]
        det_res = veh_model(
            enh, conf=min(VEHICLE_CONF, PERSON_CONF),
            iou=IOU_THRESH, classes=all_classes,
            imgsz=640, verbose=False
        )[0]

        veh_dets    = []
        person_boxes = []

        for box in det_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            if cid == COCO_PERSON_CLASS and conf >= PERSON_CONF:
                person_boxes.append((x1, y1, x2, y2))
            elif cid in COCO_VEHICLE_CLASSES and conf >= VEHICLE_CONF:
                lbl = coco_label(cid, x2-x1, y2-y1)
                veh_dets.append((x1, y1, x2, y2, conf, lbl))

        tracked_vehs  = veh_tracker.update(veh_dets, now)
        tracked_persons = pers_tracker.update(person_boxes, now)

        # ════════════════════════════════════════════════
        # STAGE 2 — Per-vehicle processing
        # ════════════════════════════════════════════════
        for det in tracked_vehs:
            vx1,vy1,vx2,vy2, vconf, vlabel, tid = det
            veh_tracker.tick(tid)
            track_info = veh_tracker.get(tid)

            is_small = vlabel in SMALL_VEHICLE_TYPES
            veh_color = VEH_COLOR_MAP.get(vlabel, (0,200,0))
            plate_text = veh_tracker.get_plate(tid)

            # ── Count this vehicle (once per track) ──
            if (track_info and
                    track_info['frame_count'] >= MIN_FRAMES_FOR_COUNT
                    and tid not in counted_veh):
                counted_veh[tid] = vlabel
                vehicle_type_counts[vlabel] += 1

            # ──────────────────────────────────────────
            # PLATE DETECTION
            # ──────────────────────────────────────────
            pad_h = int((vy2-vy1) * 0.15)   # more padding for small vehicles
            pad_w = int((vx2-vx1) * 0.08)
            rx1 = max(0, vx1 - pad_w); ry1 = max(0, vy1 - pad_h)
            rx2 = min(W, vx2 + pad_w); ry2 = min(H, vy2 + pad_h)
            roi  = enh[ry1:ry2, rx1:rx2]

            if roi.size == 0:
                continue

            # Run plate model at two inference sizes for small vehicles
            infer_sizes = [640, 1280] if is_small else [640]
            best_pbox   = None; best_pconf = 0.0

            for imsz in infer_sizes:
                p_res = plate_model(
                    roi, conf=PLATE_CONF, iou=IOU_THRESH,
                    imgsz=imsz, verbose=False
                )[0]
                for pb in p_res.boxes:
                    pc = float(pb.conf[0])
                    if pc > best_pconf:
                        best_pconf = pc
                        px1r,py1r,px2r,py2r = map(int, pb.xyxy[0])
                        best_pbox = (rx1+px1r, ry1+py1r, rx1+px2r, ry1+py2r)
                if best_pbox:
                    break   # found at first size — no need to retry

            if best_pbox:
                veh_tracker.set_plate_box(tid, best_pbox)
                apx1,apy1,apx2,apy2 = best_pbox
                area = (apx2-apx1)*(apy2-apy1)

                if area >= MIN_PLATE_AREA and plate_text is None:
                    if veh_tracker.should_submit_ocr(tid, frame_idx):
                        plate_crop = frame[max(0,apy1):apy2, max(0,apx1):apx2]
                        if plate_crop.size > 0:
                            ocr.submit(tid, plate_crop, is_small=is_small)
                            veh_tracker.mark_submitted(tid)

            # ── Collect OCR result ──
            ocr_result = ocr.get(tid)
            if ocr_result is not None:
                if ocr_result == OCRWorker.FAILED:
                    veh_tracker.mark_result(tid, success=False)
                else:
                    veh_tracker.mark_result(tid, success=True)
                    veh_tracker.set_plate(tid, ocr_result)
                    plate_text = ocr_result
                    last_t = seen_plates.get(ocr_result, 0)
                    if now - last_t >= COOLDOWN_SEC:
                        seen_plates[ocr_result] = now
                        all_plates[ocr_result]  = vlabel
                        retries = track_info['ocr_retries'] if track_info else '?'
                        print(f"  ✔  {vlabel:<12}  {ocr_result:<14}  "
                              f"conf={vconf:.2f}  retries={retries}  "
                              f"track={tid}  frame={frame_idx}")

            # ──────────────────────────────────────────
            # HELMET DETECTION (2-Wheelers only)
            # ──────────────────────────────────────────
            if vlabel == "2-Wheeler":
                vbox = (vx1, vy1, vx2, vy2)
                # Find persons whose bbox overlaps >= 30% with this vehicle
                riders = []
                for (px1,py1,px2,py2), _ in tracked_persons:
                    if box_overlap_pct((px1,py1,px2,py2), vbox) >= 0.30:
                        riders.append((px1,py1,px2,py2))

                if riders:
                    # Sort by x-position (rider is usually leftmost seat)
                    riders.sort(key=lambda b: b[0])

                    helmet_states = []
                    for i, (px1,py1,px2,py2) in enumerate(riders[:2]):
                        pcrop = frame[max(0,py1):py2, max(0,px1):px2]
                        state = helmet_det.detect(pcrop)
                        helmet_states.append(state)

                        # Accumulate stats (update each time, not just first)
                        # We only count if the track has been confirmed
                        if (track_info and
                                track_info['frame_count'] % 10 == 0):
                            helmet_stats[state] += 1

                        # Draw on person
                        role  = "Rider" if i == 0 else "Pillion"
                        clr   = (HELMET_YES if state == 'yes' else
                                 HELMET_NO  if state == 'no'  else HELMET_UNK)
                        h_lbl = (f"{role}: Helmet ✔" if state == 'yes' else
                                 f"{role}: NO Helmet" if state == 'no' else
                                 f"{role}: ?")
                        labelled_box(display, px1, py1, px2, py2, clr, h_lbl)

                    veh_tracker.set_helmet_states(tid, helmet_states)

            # ──────────────────────────────────────────
            # DRAW PLATE BOX
            # ──────────────────────────────────────────
            plate_box = veh_tracker.get_plate_box(tid) or best_pbox
            if plate_box:
                apx1,apy1,apx2,apy2 = plate_box
                if plate_text:
                    labelled_box(display, apx1,apy1,apx2,apy2,
                                  PLATE_COLOR, plate_text)
                elif track_info and track_info['ocr_pending']:
                    labelled_box(display, apx1,apy1,apx2,apy2,
                                  PEND_COLOR, "Reading…")
                elif track_info and track_info['ocr_retries'] >= MAX_OCR_RETRIES:
                    labelled_box(display, apx1,apy1,apx2,apy2,
                                  FAIL_COLOR, "Unreadable")
                else:
                    retries = track_info['ocr_retries'] if track_info else 0
                    labelled_box(display, apx1,apy1,apx2,apy2,
                                  PEND_COLOR,
                                  f"Retry {retries}/{MAX_OCR_RETRIES}")

            # ──────────────────────────────────────────
            # DRAW VEHICLE BOX
            # ──────────────────────────────────────────
            if plate_text:
                v_lbl = f"{vlabel} | {plate_text}"
            elif track_info and track_info['ocr_retries'] >= MAX_OCR_RETRIES:
                v_lbl = f"{vlabel} | ?"
            else:
                v_lbl = vlabel
            labelled_box(display, vx1,vy1,vx2,vy2, veh_color, v_lbl, thick=2)

        # ════════════════════════════════════════════════
        # HUD
        # ════════════════════════════════════════════════
        hud_lines = [
            f"FPS: {fps_disp:.1f}",
            f"Frame: {frame_idx}/{TOT}",
            f"Vehicles: {len(tracked_vehs)}",
            f"People:   {pers_tracker.count_confirmed()}",
            f"Plates:   {len(all_plates)}",
        ]
        hud_text(display, hud_lines)

        cv2.imshow("Vehicle + Plate Detector v4.0", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fn = f"screenshot_{frame_idx:06d}.jpg"
            cv2.imwrite(fn, display)
            print(f"[INFO] Saved {fn}")

    cap.release()
    cv2.destroyAllWindows()

    # ════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════
    runtime = time.time() - t_start
    print_report(all_plates, vehicle_type_counts, pers_tracker,
                 helmet_stats, runtime)


if __name__ == "__main__":
    main()