"""Configuration for traffic video analysis."""

VEHICLE_MODEL = "models/yolov8n.pt"
PLATE_MODEL = "models/license_plate_detector.pt"
CSV_PATH = "traffic_log.csv"
CSV_UPDATE_INTERVAL = 1  # seconds

CONF_THRESHOLD = 0.40
PLATE_CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
USE_HALF = True

MIN_PLATE_CHARS = 4
MAX_PLATE_CHARS = 10

MIN_TRACK_AGE = 30

# FIX: was 100 — fine for 4K since ultralytics returns original-frame coords
MIN_VEHICLE_HEIGHT_FOR_OCR = 80

COUNTING_LINE_PCT = (0.0, 0.65, 1.0, 0.65)

USER_CLASS_MAPPING = {
    "car": "Car",
    "motorcycle": "Bike/Motorcycle",
    "scooter": "Bike/Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "auto-rickshaw": "Auto Rickshaw",
    "bicycle": "Bicycle",
    "van": "Van",
}

PLATE_DETECTION_CLASSES = [
    "car",
    "truck",
    "bus",
    "auto-rickshaw",
    "motorcycle",
    "scooter",
]
NO_PLATE_CLASSES = ["bicycle"]
ALL_VEHICLE_CLASSES = [
    "car",
    "truck",
    "bus",
    "auto-rickshaw",
    "motorcycle",
    "scooter",
    "bicycle",
]

BOX_COLORS = {
    "car": (255, 0, 0),
    "truck": (0, 165, 255),
    "bus": (0, 255, 0),
    "auto-rickshaw": (0, 255, 255),
    "motorcycle": (0, 0, 255),
    "scooter": (203, 192, 255),
    "bicycle": (255, 255, 255),
}

COCO_VEHICLE_ID_MAP = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

TRACKER_CONFIG = "models/custom_bytetrack.yaml"
# FIX: reduced from 3 to 2 — more plate detection attempts per fast-moving vehicle
PLATE_DETECT_EVERY_N_FRAMES = 2
TARGET_MIN_FPS = 15
OCR_MAX_WORKERS = 4
RTSP_RECONNECT_WAIT_SEC = 5
WINDOW_NAME = "Traffic Analysis"

# --- HELMET DETECTION CONFIGS ---
HELMET_MODEL = "models/helmet_detector.pt"
HELMET_CONF_THRESHOLD = 0.45
TWO_WHEELER_CLASSES = ["motorcycle", "scooter"]
HELMET_CHECK_EVERY_N = 5
HELMET_CLASS_MAP = {
    "with_helmet": "helmet",
    "without_helmet": "no_helmet",
    "helmet": "helmet",
    "no_helmet": "no_helmet",
    "head": "no_helmet"
}

# --- PEDESTRIAN GENDER & CHILD DETECTION CONFIGS ---
GENDER_MODEL = "models/gender_detector.pt"
GENDER_CONF_THRESHOLD = 0.40
PEDESTRIAN_VEHICLE_IOU = 0.3
CHILD_HEIGHT_RATIO = 0.60
GENDER_CHECK_EVERY_N = 5
