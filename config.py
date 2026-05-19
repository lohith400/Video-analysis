"""Configuration for traffic video analysis."""

VEHICLE_MODEL = "models/yolov8n.pt"
PLATE_MODEL = "models/license_plate_detector.pt"
CSV_PATH = "traffic_log.csv"
CSV_UPDATE_INTERVAL = 1  # seconds

CONF_THRESHOLD = 0.4
PLATE_CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
USE_HALF = True

MIN_PLATE_CHARS = 4
MAX_PLATE_CHARS = 10

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

TRACKER_CONFIG = "bytetrack.yaml"
PLATE_DETECT_EVERY_N_FRAMES = 3
TARGET_MIN_FPS = 15
OCR_MAX_WORKERS = 4
RTSP_RECONNECT_WAIT_SEC = 5
WINDOW_NAME = "Traffic Analysis"
