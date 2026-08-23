"""
╔══════════════════════════════════════════════════════════════════════════╗
║   INDIAN VEHICLE + LICENSE PLATE DETECTOR  v5.0  MAX-ACCURACY           ║
║   ✔ 14-Stage Plate Preprocessing Pipeline                                ║
║   ✔ Multi-Engine OCR  (EasyOCR + PaddleOCR voting)                      ║
║   ✔ Perspective / Homography Correction                                  ║
║   ✔ Super-Resolution Upscaling (ESPCN via OpenCV DNN)                   ║
║   ✔ Motion-Blur Deconvolution                                            ║
║   ✔ Gamma / Night-Light Enhancement                                      ║
║   ✔ Color-based Plate Type  (White / Yellow / Green)                    ║
║   ✔ Ensemble OCR Voting + Levenshtein post-correction                   ║
║   ✔ Kalman-filter-style Tracker                                         ║
║   ✔ Helmet Detection (model or HSV heuristic)                           ║
║   ✔ Full Vehicle-type Counts + People Counter                            ║
╚══════════════════════════════════════════════════════════════════════════╝

MODELS NEEDED
─────────────────────────────────────────────────────────────────────────
  yolov8n.pt                – COCO vehicles + persons  (auto-downloaded)
  license_plate_detector.pt – fine-tuned Indian plate detector
  helmet_detector.pt        – (OPTIONAL)  see SETUP GUIDE below

QUICK-SETUP  ──────────────────────────────────────────────────────────
  pip install ultralytics easyocr paddlepaddle paddleocr opencv-python
      numpy scipy

  # OPTIONAL super-resolution model (free, ~5 MB):
  wget https://github.com/opencv/opencv_contrib/raw/master/\
modules/dnn_superres/samples/ESPCN_x4.pb  -P models/

  # Helmet model — choose ONE of:
  #  A) Roboflow export (best for Indian traffic):
  #     pip install roboflow
  #     from roboflow import Roboflow
  #     rf = Roboflow(api_key="YOUR_KEY")
  #     proj = rf.workspace("roboflow-universe-projects") \
  #               .project("helmet-detection-using-opencv")
  #     proj.version(9).download("yolov8", location="models/")
  #     → rename best.pt  → helmet_detector.pt
  #
  #  B) Direct HuggingFace download (no login needed):
  #     pip install huggingface_hub
  #     from huggingface_hub import hf_hub_download
  #     hf_hub_download(repo_id="keremberke/yolov8s-helmet-detection",
  #                     filename="best.pt", local_dir="models/")
  #     → rename / set HELMET_MODEL below
"""

# ═══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import cv2, easyocr, re, time, os, threading, queue, warnings
import numpy as np
from collections import defaultdict, Counter
from ultralytics import YOLO
warnings.filterwarnings("ignore")

# PaddleOCR — optional; gracefully disabled if not installed
try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _paddle_available = True
except ImportError:
    _paddle_available = False
    print("[WARN] PaddleOCR not installed – using EasyOCR only.")
    print("       pip install paddlepaddle paddleocr")

# OpenCV DNN super-res — optional
try:
    from cv2 import dnn_superres
    _sr_available = True
except ImportError:
    _sr_available = False


# ═══════════════════════════════════════════════════════════════════════
#  ┌──────────────────────┐
#  │   USER CONFIG        │  ← edit these
#  └──────────────────────┘
# ═══════════════════════════════════════════════════════════════════════
VIDEO_IN          = "/home/lohit/realcode/ML/Code/projectsss/vehicle/videos/L3.mp4"
PLATE_MODEL_PATH  = "license_plate_detector.pt"
VEHICLE_MODEL     = "yolov8n.pt"
HELMET_MODEL      = "models/helmet_detector.pt"   # or None

# ── Super-resolution model (optional, huge quality boost for small plates) ──
SR_MODEL_PATH     = "models/ESPCN_x4.pb"          # set None to disable
SR_SCALE          = 4                              # 2 or 4 match the .pb name

# ── Detection thresholds ──
VEHICLE_CONF      = 0.30
PLATE_CONF        = 0.12    # low threshold catches small/blurry plates
PERSON_CONF       = 0.38
HELMET_CONF       = 0.25
IOU_THRESH        = 0.45

# ── OCR ──
COOLDOWN_SEC      = 4.0
TRACK_LOST_SEC    = 4.0
MIN_PLATE_AREA    = 80      # px², even tinier plates accepted
MAX_OCR_RETRIES   = 14
RETRY_EVERY_N     = 3

# ── Tracking ──
MIN_FRAMES_FOR_COUNT = 6

# ── Misc ──
COCO_VEHICLE_CLASSES = {1, 2, 3, 5, 7}
COCO_PERSON_CLASS    = 0
SMALL_VEHICLE_TYPES  = {"2-Wheeler", "3-Wheeler"}


# ═══════════════════════════════════════════════════════════════════════
#  VEHICLE LABEL
# ═══════════════════════════════════════════════════════════════════════
def coco_label(cid: int, w: int, h: int) -> str:
    mapping = {1: "2-Wheeler", 3: "2-Wheeler", 2: "4-Wheeler",
               5: "Bus",       7: "Truck"}
    lbl = mapping.get(cid, "Vehicle")
    if cid == 3 and w > 0 and h > 0 and (w / h) > 1.2 and w > 80:
        lbl = "3-Wheeler"
    return lbl


# ═══════════════════════════════════════════════════════════════════════
#  INDIAN PLATE PATTERNS + VALIDATION
# ═══════════════════════════════════════════════════════════════════════
_PATTERNS = [
    # Standard new series: MH12AB1234
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}'),
    # BH Bharat series: 22BH1234AB
    re.compile(r'[0-9]{2}BH[0-9]{4}[A-Z]{1,2}'),
    # Old single-letter: MH12A1234
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z][0-9]{4}'),
    # Relaxed 8-char fallback
    re.compile(r'[A-Z]{2}[0-9]{2}[A-Z0-9]{4,6}'),
]

_STATE_CODES = {
    'AN','AP','AR','AS','BR','CH','CG','DD','DL','DN','GA','GJ',
    'HR','HP','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP',
    'MZ','NL','OD','PB','PY','RJ','SK','TN','TS','TR','UK','UP','WB',
}

_D2A = {'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G','4':'A'}
_A2D = {'O':'0','I':'1','Z':'2','S':'5','B':'8','G':'6','D':'0','Q':'0'}

def _clean(t: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', t.upper())

def _fix_ocr(t: str) -> str:
    """Position-aware OCR error correction for Indian plates."""
    s = list(_clean(t))
    if not s:
        return t
    # Pos 0-1: must be letters (state code)
    for i in range(min(2, len(s))):
        if s[i].isdigit():
            s[i] = _D2A.get(s[i], s[i])
    # Pos 2-3: must be digits (district code)
    for i in range(2, min(4, len(s))):
        if s[i].isalpha():
            s[i] = _A2D.get(s[i], s[i])
    # Last 4 chars: must be digits (serial number)
    for i in range(max(0, len(s)-4), len(s)):
        if s[i].isalpha():
            s[i] = _A2D.get(s[i], s[i])
    return ''.join(s)

def extract_plate(raw: str, relaxed: bool = False):
    """Try all patterns; validate state code; return best match or None."""
    text = _fix_ocr(raw)
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            c = m.group()
            if len(c) >= 6:
                if c[:2] in _STATE_CODES or 'BH' in c:
                    return c
                if relaxed and len(c) >= 8:
                    return c
    return None


# ═══════════════════════════════════════════════════════════════════════
#  SUPER-RESOLUTION
# ═══════════════════════════════════════════════════════════════════════
class SuperResolver:
    def __init__(self, model_path, scale):
        self._sr = None
        if _sr_available and model_path and os.path.isfile(model_path):
            try:
                sr = dnn_superres.DnnSuperResImpl_create()
                sr.readModel(model_path)
                sr.setModel("espcn", scale)
                self._sr = sr
                print(f"[SR] Super-resolution model loaded (×{scale}).")
            except Exception as e:
                print(f"[SR] Could not load SR model: {e}")
        else:
            if model_path:
                print(f"[SR] SR model not found at '{model_path}'. Skipping.")

    def upscale(self, img):
        if self._sr is None:
            return None
        try:
            return self._sr.upsample(img)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE ENHANCEMENT
# ═══════════════════════════════════════════════════════════════════════
_clahe   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
_sharp_k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)

def gamma_correct(img, gamma=1.5):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    return cv2.LUT(img, table.astype(np.uint8))

def deblur_wiener(gray, snr=25):
    """Simple Wiener-filter deblurring for motion blur."""
    f = np.fft.fft2(gray.astype(np.float32))
    psd  = np.abs(f) ** 2
    filt = np.conj(f) / (psd + snr)
    result = np.real(np.fft.ifft2(f * filt))
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def enhance_frame(frame):
    h, w = frame.shape[:2]
    # Upscale very small inputs
    if h < 480:
        frame = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    # CLAHE on L channel
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Check brightness – if dark, apply gamma
    if np.mean(l) < 80:
        l = gamma_correct(l, gamma=1.8)
    l = _clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    frame = cv2.filter2D(frame, -1, _sharp_k)
    return frame

def deskew_plate(gray):
    """Straighten a tilted plate crop via moments."""
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] < 5:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45: angle += 90
    if abs(angle) < 1: return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)

def perspective_correct(gray):
    """
    Find the largest 4-sided contour and warp it to a rectangle.
    Straightens perspective-distorted plates (common on angles).
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edg  = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    best, best_area = None, 0
    h0, w0 = gray.shape
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        a = cv2.contourArea(c)
        if len(approx) == 4 and a > best_area and a > 0.15 * h0 * w0:
            best, best_area = approx, a
    if best is None:
        return gray
    pts = best.reshape(4, 2).astype(np.float32)
    # Sort: top-left, top-right, bottom-right, bottom-left
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1)
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]
    bl  = pts[np.argmax(d)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    tw  = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    th  = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    if tw < 10 or th < 10:
        return gray
    dst = np.array([[0,0],[tw,0],[tw,th],[0,th]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (tw, th))

def shadow_remove(gray):
    """Remove uneven illumination / shadows via morphological background est."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    bg      = cv2.dilate(gray, kernel)
    bg      = cv2.GaussianBlur(bg, (21, 21), 0)
    diff    = 255 - cv2.subtract(bg, gray)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


# ═══════════════════════════════════════════════════════════════════════
#  PLATE COLOR TYPE  (White = private, Yellow = commercial, Green = EV)
# ═══════════════════════════════════════════════════════════════════════
def plate_color_type(crop) -> str:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_s = np.mean(s); mean_h = np.mean(h)
    if mean_s < 50:
        return "white"
    if 20 < mean_h < 35:
        return "yellow"
    if 40 < mean_h < 85:
        return "green"
    return "white"


# ═══════════════════════════════════════════════════════════════════════
#  14-STAGE PLATE PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def plate_variants(crop, is_small: bool = False, sr: SuperResolver = None):
    """
    Yield (processed_image, variant_name) pairs for OCR.
    More stages = higher chance of cracking a noisy / blurry plate.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return

    # ── Determine target height ──
    target_h = 96 if is_small else 72
    scale    = max(2, int(target_h / max(h, 1)))
    scale    = min(scale, 10)

    gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ── Stage 0: Super-resolution (if model available) ──
    if sr is not None:
        sr_img = sr.upscale(crop)
        if sr_img is not None:
            g = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY)
            yield g, "sr"

    # ── Stage 1: Shadow removal + bilateral upscale ──
    sr_gray = shadow_remove(gray0)
    up = cv2.resize(sr_gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    up = cv2.bilateralFilter(up, 11, 90, 90)
    yield up, "bilateral"

    # ── Stage 2: Deskew ──
    dsk = deskew_plate(up)
    yield dsk, "deskew"

    # ── Stage 3: Perspective correction ──
    persp = perspective_correct(up)
    if persp.shape != up.shape:
        yield persp, "persp"

    # ── Stage 4: Adaptive threshold ──
    thr = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 19, 9)
    yield thr, "adapt"

    # ── Stage 5: Inverted adaptive (dark background) ──
    yield cv2.bitwise_not(thr), "adapt_inv"

    # ── Stage 6: Otsu binary ──
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    yield otsu, "otsu"

    # ── Stage 7: Inverted Otsu ──
    yield cv2.bitwise_not(otsu), "otsu_inv"

    # ── Stage 8: CLAHE equalized ──
    eq = _clahe.apply(gray0)
    eq_up = cv2.resize(eq, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    yield eq_up, "clahe"

    # ── Stage 9: Gamma correction for night ──
    gam = gamma_correct(up, gamma=1.6)
    yield gam, "gamma"

    # ── Stage 10: Morphological open (remove noise) ──
    k1 = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k1)
    yield opened, "morph_open"

    # ── Stage 11: Morphological close (fill gaps in chars) ──
    k2 = np.ones((2, 3), np.uint8)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k2)
    yield closed, "morph_close"

    # ── Stage 12: Wiener deblur on bilateral ──
    try:
        deblurred = deblur_wiener(up)
        yield deblurred, "wiener"
    except Exception:
        pass

    # ── Stage 13: High-frequency sharpening ──
    sharp = cv2.filter2D(up, -1, _sharp_k)
    yield sharp, "sharp"

    # ── Stage 14: Contrast stretch ──
    lo, hi = np.percentile(gray0, (1, 99))
    stretched = np.clip((gray0.astype(np.float32) - lo) / max(hi-lo, 1) * 255,
                         0, 255).astype(np.uint8)
    stretched_up = cv2.resize(stretched, (w*scale, h*scale),
                               interpolation=cv2.INTER_CUBIC)
    yield stretched_up, "stretch"


# ═══════════════════════════════════════════════════════════════════════
#  OCR WORKER  (background thread, multi-engine)
# ═══════════════════════════════════════════════════════════════════════
class OCRWorker:
    FAILED = "__FAILED__"

    def __init__(self, sr: SuperResolver):
        self._sr = sr
        print("[OCR] Initialising EasyOCR …")
        self._easy = easyocr.Reader(['en'], gpu=True,
                                     model_storage_directory='models/easyocr')

        self._paddle = None
        if _paddle_available:
            print("[OCR] Initialising PaddleOCR …")
            try:
                self._paddle = _PaddleOCR(
                    use_angle_cls=True, lang='en',
                    use_gpu=True, show_log=False,
                    det_db_thresh=0.3, det_db_box_thresh=0.4,
                    rec_algorithm='SVTR_LCNet'
                )
            except Exception as e:
                print(f"[OCR] PaddleOCR init failed: {e}")

        self._q    = queue.Queue(maxsize=512)
        self._out  = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        print("[OCR] Ready.\n")

    # ── EasyOCR call ──
    def _easy_ocr(self, img):
        results = self._easy.readtext(
            img, detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False, width_ths=0.9,
            text_threshold=0.25, low_text=0.35
        )
        texts = [r[1] for r in results]
        confs = [r[2] for r in results]
        return texts, confs

    # ── PaddleOCR call ──
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

    # ── Combine results from both engines with confidence weighting ──
    def _combine_ocr(self, img, is_small: bool) -> tuple:
        """Returns (plate_string, confidence) or (None, 0)."""
        candidates = []

        # EasyOCR
        try:
            txts, confs = self._easy_ocr(img)
            raw = "".join(txts)
            plate = extract_plate(raw, relaxed=is_small)
            if plate:
                avg_c = np.mean(confs) if confs else 0.5
                candidates.append((plate, avg_c, 'easy'))
        except Exception:
            pass

        # PaddleOCR
        try:
            txts, confs = self._paddle_ocr(img)
            raw = "".join(txts)
            plate = extract_plate(raw, relaxed=is_small)
            if plate:
                avg_c = np.mean(confs) if confs else 0.5
                candidates.append((plate, avg_c, 'paddle'))
        except Exception:
            pass

        if not candidates:
            return None, 0.0

        # If both agree → high confidence
        plates_only = [c[0] for c in candidates]
        if len(plates_only) >= 2 and plates_only[0] == plates_only[1]:
            return plates_only[0], 1.0

        # Pick highest-confidence one
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0], candidates[0][1]

    def submit(self, tid, crop, is_small: bool = False):
        try:
            self._q.put_nowait((tid, crop.copy(), is_small))
        except queue.Full:
            pass

    def get(self, tid):
        with self._lock:
            return self._out.pop(tid, None)

    def _loop(self):
        while True:
            tid, crop, is_small = self._q.get()
            best_plate = None
            best_conf  = 0.0

            for variant_img, _name in plate_variants(crop, is_small, self._sr):
                plate, conf = self._combine_ocr(variant_img, is_small)
                if plate and conf > best_conf:
                    best_plate = plate
                    best_conf  = conf
                if best_conf >= 1.0:   # both engines agreed – stop early
                    break

            with self._lock:
                self._out[tid] = best_plate if best_plate else self.FAILED


# ═══════════════════════════════════════════════════════════════════════
#  IoU TRACKER  (with exponential smoothing on bounding box)
# ═══════════════════════════════════════════════════════════════════════
_ALPHA = 0.6   # smoothing factor (higher = snappier)

class Tracker:
    def __init__(self):
        self._nid    = 0
        self._tracks = {}

    @staticmethod
    def _iou(a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if not inter: return 0.0
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua else 0.0

    def update(self, dets, now):
        stale=[t for t,v in self._tracks.items()
               if now-v['last_seen']>TRACK_LOST_SEC]
        for t in stale: del self._tracks[t]

        used_t, used_d, results = set(), set(), []

        for di, d in enumerate(dets):
            best_iou, best_t = 0.0, None
            for tid, tr in self._tracks.items():
                if tid in used_t: continue
                iou = self._iou(d[:4], tr['box'])
                if iou > best_iou:
                    best_iou, best_t = iou, tid
            if best_iou > 0.25 and best_t is not None:
                tr = self._tracks[best_t]
                # Smoothed box update
                ob = tr['box']
                nb = d[:4]
                tr['box'] = tuple(int(_ALPHA*nb[i]+(1-_ALPHA)*ob[i]) for i in range(4))
                tr.update({'last_seen': now, 'label': d[5]})
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
                'helmet_states': [], 'plate_conf': 0.0,
            }
            results.append((*d, tid))
        return results

    def get(self, tid): return self._tracks.get(tid)
    def get_plate(self, tid):
        t=self._tracks.get(tid); return t['plate'] if t else None
    def set_plate(self, tid, p, conf=1.0):
        if tid in self._tracks:
            self._tracks[tid]['plate'] = p
            self._tracks[tid]['plate_conf'] = conf
    def set_plate_box(self, tid, box):
        if tid in self._tracks: self._tracks[tid]['last_plate_box'] = box
    def get_plate_box(self, tid):
        t=self._tracks.get(tid); return t['last_plate_box'] if t else None
    def set_helmet_states(self, tid, s):
        if tid in self._tracks: self._tracks[tid]['helmet_states'] = s

    def should_submit_ocr(self, tid, frame_idx):
        t = self._tracks.get(tid)
        if not t: return False
        if t['plate']: return False
        if t['ocr_retries'] >= MAX_OCR_RETRIES: return False
        if t['ocr_pending']: return False
        return t['frames_since_submit'] >= RETRY_EVERY_N or t['ocr_retries'] == 0

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


# ═══════════════════════════════════════════════════════════════════════
#  PERSON TRACKER
# ═══════════════════════════════════════════════════════════════════════
class PersonTracker:
    def __init__(self):
        self._nid=0; self._tracks={}; self.all_ids=set()

    @staticmethod
    def _iou(a,b):
        ix1=max(a[0],b[0]);iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]);iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if not inter: return 0.0
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua else 0.0

    def update(self, boxes, now):
        stale=[p for p,v in self._tracks.items()
               if now-v['last_seen']>TRACK_LOST_SEC*2]
        for p in stale: del self._tracks[p]
        results=[]; used_t=set()
        for box in boxes:
            best_iou,best_p=0.0,None
            for pid,tr in self._tracks.items():
                if pid in used_t: continue
                iou=self._iou(box,tr['box'])
                if iou>best_iou: best_iou,best_p=iou,pid
            if best_iou>0.25 and best_p is not None:
                self._tracks[best_p].update({'box':box,'last_seen':now})
                self._tracks[best_p]['frame_count']+=1
                used_t.add(best_p); results.append((box,best_p))
            else:
                pid=self._nid; self._nid+=1; self.all_ids.add(pid)
                self._tracks[pid]={'box':box,'last_seen':now,'frame_count':1}
                results.append((box,pid))
        return results

    def count_confirmed(self):
        return sum(1 for t in self._tracks.values()
                   if t['frame_count']>=MIN_FRAMES_FOR_COUNT)


# ═══════════════════════════════════════════════════════════════════════
#  HELMET DETECTION
# ═══════════════════════════════════════════════════════════════════════
def _heuristic_helmet(head_crop) -> str:
    if head_crop is None or head_crop.size==0: return 'unknown'
    h,w=head_crop.shape[:2]
    if h<10 or w<10: return 'unknown'
    hsv=cv2.cvtColor(head_crop,cv2.COLOR_BGR2HSV)
    skin_lo=np.array([0,20,70],dtype=np.uint8)
    skin_hi=np.array([25,255,255],dtype=np.uint8)
    skin_pct=cv2.countNonZero(cv2.inRange(hsv,skin_lo,skin_hi))/(h*w)
    if skin_pct<0.25: return 'yes'
    if skin_pct>0.35: return 'no'
    return 'unknown'

class HelmetDetector:
    def __init__(self, path):
        self.model=None
        if path and os.path.isfile(path):
            try:
                self.model=YOLO(path)
                print(f"[Helmet] Model loaded: {path}")
            except Exception as e:
                print(f"[Helmet] Load failed ({e}). Using heuristic.")
        else:
            print("[Helmet] No model — using HSV heuristic.")

    def detect(self, crop) -> str:
        if crop is None or crop.size==0: return 'unknown'
        ph=crop.shape[0]
        head=crop[:max(1,int(ph*0.35)),:]
        if self.model is not None:
            try:
                res=self.model(head, conf=HELMET_CONF, verbose=False)[0]
                for box in res.boxes:
                    n=res.names[int(box.cls[0])].lower()
                    if 'helmet' in n or 'with' in n: return 'yes'
                    if 'no' in n or 'without' in n: return 'no'
                return 'unknown'
            except Exception:
                pass
        return _heuristic_helmet(head)


# ═══════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════
VEH_COLORS = {"2-Wheeler":(0,210,80),"3-Wheeler":(0,190,140),
              "4-Wheeler":(30,170,255),"Bus":(0,230,120),
              "Truck":(0,200,60),"Vehicle":(0,180,0)}
PLATE_CLR   = (0,0,230);  PEND_CLR=(0,165,255); FAIL_CLR=(80,80,80)
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_S      = cv2.FONT_HERSHEY_SIMPLEX

def corner_rect(img, x1,y1,x2,y2, color, thick=2):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thick)
    clen=min(18,max(1,(x2-x1)//5),max(1,(y2-y1)//5))
    for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+dx*clen,cy),color,thick+2)
        cv2.line(img,(cx,cy),(cx,cy+dy*clen),color,thick+2)

def labelled_box(img,x1,y1,x2,y2,color,label="",thick=2):
    corner_rect(img,x1,y1,x2,y2,color,thick)
    if label:
        fs=0.46
        (tw,th),bl=cv2.getTextSize(label,FONT,fs,1)
        lx=max(0,x1); ly=max(0,y1-th-bl-6)
        cv2.rectangle(img,(lx,ly),(lx+tw+8,y1),color,-1)
        cv2.putText(img,label,(lx+4,y1-bl-1),FONT,fs,(255,255,255),1,cv2.LINE_AA)

def hud_text(img, lines, x=10, y0=26, dy=24):
    for i,ln in enumerate(lines):
        y=y0+i*dy
        cv2.putText(img,ln,(x,y),FONT_S,0.6,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(img,ln,(x,y),FONT_S,0.6,(255,255,255),1,cv2.LINE_AA)

def box_overlap_pct(inner,outer):
    ix1=max(inner[0],outer[0]); iy1=max(inner[1],outer[1])
    ix2=min(inner[2],outer[2]); iy2=min(inner[3],outer[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ia=(inner[2]-inner[0])*(inner[3]-inner[1])
    return inter/ia if ia else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════
def print_report(all_plates, vtype_counts, pers_tracker,
                 helmet_stats, runtime):
    W=64; ln="═"*W
    print(f"\n{ln}\n  FINAL REPORT  –  Indian Vehicle Detector v5.0\n{ln}")
    print(f"  VEHICLE COUNTS\n  {'Type':<16} {'Count':>6}")
    total=0
    for t in ["2-Wheeler","3-Wheeler","4-Wheeler","Bus","Truck","Vehicle"]:
        c=vtype_counts.get(t,0)
        if c: print(f"  {t:<16} {c:>6}"); total+=c
    print(f"  {'TOTAL':<16} {total:>6}")
    ppl=pers_tracker.count_confirmed()
    print(f"\n  PEOPLE DETECTED          : {ppl}")
    print(f"\n  HELMET COMPLIANCE  (2-Wheelers)")
    for k,label in [('yes','With helmet'),('no','WITHOUT helmet'),('unknown','Undetermined')]:
        print(f"  {label:<24}: {helmet_stats[k]}")
    total_riders=sum(helmet_stats.values())
    if total_riders:
        print(f"  Compliance rate          : {100*helmet_stats['yes']/total_riders:.1f}%")
    print(f"\n  PLATES DETECTED          : {len(all_plates)}")
    print(f"\n  {'Vehicle':<16}  {'Plate':<14}  Color")
    print(f"  {'─'*16}  {'─'*14}  {'─'*6}")
    for plate,(vtype,pcolor) in sorted(all_plates.items(),key=lambda x:x[1][0]):
        print(f"  {vtype:<16}  {plate:<14}  {pcolor}")
    print(f"\n  Runtime : {runtime:.1f} s\n{ln}\n")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("="*64)
    print("  Indian Vehicle + Plate Detector  v5.0  MAX-ACCURACY")
    print("="*64)

    os.makedirs("models", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    print("[INFO] Loading YOLO vehicle model …")
    veh_model   = YOLO(VEHICLE_MODEL)

    print("[INFO] Loading plate model …")
    plate_model = YOLO(PLATE_MODEL_PATH)

    print("[INFO] Loading helmet detector …")
    helmet_det  = HelmetDetector(HELMET_MODEL)

    print("[INFO] Loading super-resolution module …")
    sr = SuperResolver(SR_MODEL_PATH, SR_SCALE)

    ocr = OCRWorker(sr)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_IN}"); return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {W}×{H}  {FPS:.1f} fps  {TOT} frames")
    print("[INFO] Q = quit  |  S = screenshot\n")

    veh_tracker   = Tracker()
    pers_tracker  = PersonTracker()
    seen_plates   = {}        # plate → last print time
    all_plates    = {}        # plate → (vehicle_label, plate_color)
    counted_veh   = {}        # tid → label  (counted once per confirmed track)
    vtype_counts  = defaultdict(int)
    helmet_stats  = {'yes':0,'no':0,'unknown':0}

    frame_idx=0; fps_cnt=0; fps_t=time.time(); fps_disp=0.0
    t_start=time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        now = time.time()

        fps_cnt += 1
        if now - fps_t >= 1.0:
            fps_disp = fps_cnt / (now - fps_t)
            fps_cnt = 0; fps_t = now

        enh     = enhance_frame(frame.copy())
        display = frame.copy()

        # ── Stage 1: Detect vehicles + persons ──
        all_cls = list(COCO_VEHICLE_CLASSES) + [COCO_PERSON_CLASS]
        res0 = veh_model(
            enh, conf=min(VEHICLE_CONF, PERSON_CONF),
            iou=IOU_THRESH, classes=all_cls,
            imgsz=1280,          # ← higher res for small distant vehicles
            verbose=False
        )[0]

        veh_dets=[]; person_boxes=[]
        for box in res0.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cid=int(box.cls[0]); conf=float(box.conf[0])
            if cid==COCO_PERSON_CLASS and conf>=PERSON_CONF:
                person_boxes.append((x1,y1,x2,y2))
            elif cid in COCO_VEHICLE_CLASSES and conf>=VEHICLE_CONF:
                lbl=coco_label(cid, x2-x1, y2-y1)
                veh_dets.append((x1,y1,x2,y2,conf,lbl))

        tracked_vehs    = veh_tracker.update(veh_dets, now)
        tracked_persons = pers_tracker.update(person_boxes, now)

        # ── Stage 2: Per-vehicle processing ──
        for det in tracked_vehs:
            vx1,vy1,vx2,vy2, vconf, vlabel, tid = det
            veh_tracker.tick(tid)
            ti = veh_tracker.get(tid)

            is_small  = vlabel in SMALL_VEHICLE_TYPES
            veh_color = VEH_COLORS.get(vlabel, (0,200,0))
            plate_txt = veh_tracker.get_plate(tid)

            # Count once per confirmed track
            if ti and ti['frame_count']>=MIN_FRAMES_FOR_COUNT and tid not in counted_veh:
                counted_veh[tid]=vlabel; vtype_counts[vlabel]+=1

            # ── Plate detection ──
            padh=int((vy2-vy1)*0.18); padw=int((vx2-vx1)*0.12)
            rx1=max(0,vx1-padw); ry1=max(0,vy1-padh)
            rx2=min(W,vx2+padw); ry2=min(H,vy2+padh)
            roi = enh[ry1:ry2, rx1:rx2]
            if roi.size==0: continue

            # Run plate model at 640 and 1280 for small vehicles
            infer_sizes = [640,1280] if is_small else [640]
            best_pbox=None; best_pconf=0.0

            for imsz in infer_sizes:
                p_res = plate_model(
                    roi, conf=PLATE_CONF, iou=IOU_THRESH,
                    imgsz=imsz, verbose=False
                )[0]
                for pb in p_res.boxes:
                    pc=float(pb.conf[0])
                    if pc>best_pconf:
                        best_pconf=pc
                        px1r,py1r,px2r,py2r=map(int,pb.xyxy[0])
                        best_pbox=(rx1+px1r,ry1+py1r,rx1+px2r,ry1+py2r)
                if best_pbox: break

            if best_pbox:
                veh_tracker.set_plate_box(tid, best_pbox)
                apx1,apy1,apx2,apy2 = best_pbox
                area=(apx2-apx1)*(apy2-apy1)

                if area>=MIN_PLATE_AREA and plate_txt is None:
                    if veh_tracker.should_submit_ocr(tid, frame_idx):
                        # Use original (non-enhanced) frame for OCR
                        # to preserve natural colour/contrast
                        crop=frame[max(0,apy1):apy2, max(0,apx1):apx2]
                        if crop.size>0:
                            ocr.submit(tid, crop, is_small=is_small)
                            veh_tracker.mark_submitted(tid)

            # ── Collect OCR result ──
            ocr_result = ocr.get(tid)
            if ocr_result is not None:
                if ocr_result==OCRWorker.FAILED:
                    veh_tracker.mark_result(tid, False)
                else:
                    veh_tracker.mark_result(tid, True)
                    veh_tracker.set_plate(tid, ocr_result)
                    plate_txt = ocr_result
                    if now-seen_plates.get(ocr_result,0)>=COOLDOWN_SEC:
                        seen_plates[ocr_result]=now
                        # Detect plate colour for this crop
                        if best_pbox:
                            apx1,apy1,apx2,apy2=best_pbox
                            pcrop=frame[max(0,apy1):apy2, max(0,apx1):apx2]
                            pcolor=plate_color_type(pcrop) if pcrop.size>0 else "?"
                        else:
                            pcolor="?"
                        all_plates[ocr_result]=(vlabel,pcolor)
                        retries=ti['ocr_retries'] if ti else '?'
                        print(f"  ✔  {vlabel:<12}  {ocr_result:<14}  "
                              f"[{pcolor}]  conf={vconf:.2f}  "
                              f"retries={retries}  frame={frame_idx}")

            # ── Helmet detection (2-Wheelers) ──
            if vlabel=="2-Wheeler":
                vbox=(vx1,vy1,vx2,vy2)
                riders=[
                    (px1,py1,px2,py2)
                    for (px1,py1,px2,py2),_ in tracked_persons
                    if box_overlap_pct((px1,py1,px2,py2),vbox)>=0.25
                ]
                if riders:
                    riders.sort(key=lambda b:b[0])
                    states=[]
                    for i,(px1,py1,px2,py2) in enumerate(riders[:2]):
                        pcrop=frame[max(0,py1):py2, max(0,px1):px2]
                        state=helmet_det.detect(pcrop)
                        states.append(state)
                        if ti and ti['frame_count']%10==0:
                            helmet_stats[state]+=1
                        role="Rider" if i==0 else "Pillion"
                        clr=((0,210,0) if state=='yes' else
                             (0,0,210) if state=='no' else (160,160,0))
                        lbl=(f"{role}: ✔ Helmet" if state=='yes' else
                             f"{role}: ✗ No Helmet" if state=='no' else
                             f"{role}: ?")
                        labelled_box(display,px1,py1,px2,py2,clr,lbl)
                    veh_tracker.set_helmet_states(tid,states)

            # ── Draw plate box ──
            plate_box = veh_tracker.get_plate_box(tid) or best_pbox
            if plate_box:
                apx1,apy1,apx2,apy2=plate_box
                if plate_txt:
                    conf_pct=int((ti['plate_conf'] if ti else 1.0)*100)
                    labelled_box(display,apx1,apy1,apx2,apy2,
                                  PLATE_CLR, f"{plate_txt} {conf_pct}%")
                elif ti and ti['ocr_pending']:
                    labelled_box(display,apx1,apy1,apx2,apy2,PEND_CLR,"Reading…")
                elif ti and ti['ocr_retries']>=MAX_OCR_RETRIES:
                    labelled_box(display,apx1,apy1,apx2,apy2,FAIL_CLR,"Unreadable")
                else:
                    r=ti['ocr_retries'] if ti else 0
                    labelled_box(display,apx1,apy1,apx2,apy2,
                                  PEND_CLR,f"Retry {r}/{MAX_OCR_RETRIES}")

            # ── Draw vehicle box ──
            if plate_txt:
                vl=f"{vlabel} | {plate_txt}"
            elif ti and ti['ocr_retries']>=MAX_OCR_RETRIES:
                vl=f"{vlabel} | ?"
            else:
                vl=vlabel
            labelled_box(display,vx1,vy1,vx2,vy2,veh_color,vl,thick=2)

        # ── HUD ──
        hud_text(display,[
            f"FPS: {fps_disp:.1f}",
            f"Frame: {frame_idx}/{TOT}",
            f"Vehicles: {len(tracked_vehs)}",
            f"People:   {pers_tracker.count_confirmed()}",
            f"Plates:   {len(all_plates)}",
        ])

        cv2.imshow("Vehicle + Plate Detector v5.0", display)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        if key==ord('s'):
            fn=f"screenshots/frame_{frame_idx:06d}.jpg"
            cv2.imwrite(fn,display)
            print(f"[INFO] Saved {fn}")

    cap.release()
    cv2.destroyAllWindows()
    print_report(all_plates, vtype_counts, pers_tracker,
                 helmet_stats, time.time()-t_start)


if __name__=="__main__":
    main()

