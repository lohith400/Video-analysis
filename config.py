"""Configuration for traffic video analysis."""

VEHICLE_MODEL = "models/yolov8n.pt"
PLATE_MODEL = "models/license_plate_detector.pt"
CSV_PATH = "traffic_log.csv"
CSV_UPDATE_INTERVAL = 1  # seconds

CONF_THRESHOLD = 0.25  # Lowered from 0.40 to capture small/distant vehicles; tracker handles noise
PLATE_CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
USE_HALF = True

MIN_PLATE_CHARS = 4
MAX_PLATE_CHARS = 10

# Minimum active frames a track must exist to filter out ephemeral false positives
MIN_TRACK_AGE = 4

# Virtual counting line as percentages: (start_x_pct, start_y_pct, end_x_pct, end_y_pct)
# Placed horizontally at 65% height
COUNTING_LINE_PCT = (0.0, 0.65, 1.0, 0.65)

# Map raw detection classes (COCO/Custom) to precise target user classes
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

# COCO class IDs from yolov8n (when using default pretrained weights)
COCO_VEHICLE_ID_MAP = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

TRACKER_CONFIG = "models/custom_bytetrack.yaml"
PLATE_DETECT_EVERY_N_FRAMES = 3
TARGET_MIN_FPS = 15
OCR_MAX_WORKERS = 4
RTSP_RECONNECT_WAIT_SEC = 5
WINDOW_NAME = "Traffic Analysis"
