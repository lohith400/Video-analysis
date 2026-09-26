"""Configuration for real-time traffic video analysis and ANPR pipeline."""

# Model Paths
VEHICLE_MODEL = "models/yolov8n.pt"
PLATE_MODEL = "models/v2.0/license_plate_detector.pt"
HELMET_MODEL = "models/v2.0/helmet_detector.pt"
GENDER_MODEL = "models/gender_detector.pt"
TRACKER_CONFIG = "models/custom_bytetrack.yaml"

# Output Logging Settings
CSV_PATH = "traffic_log.csv"
CSV_UPDATE_INTERVAL = 1.0  # seconds

# Vehicle Detection & Tracking Thresholds
CONF_THRESHOLD = 0.35  # Balanced sensitivity for dense traffic flows
IOU_THRESHOLD = 0.45
USE_HALF = True  # FP16 acceleration for vehicle YOLO model
MIN_TRACK_AGE = 5  # Reduced from 30: Vector line-crossing handles zero duplicate counts mathematically

# License Plate Detection & OCR Tuning
PLATE_CONF_THRESHOLD = 0.20  # Lowered to capture small, fast, or angled plates on two-wheelers
PLATE_USE_HALF = False  # Keep FP32 to avoid fused Conv+BN graph issues on older checkpoints
PLATE_DETECT_EVERY_N_FRAMES = 1  # Synchronous per-frame detection for maximum recall
MIN_VEHICLE_HEIGHT_FOR_OCR = 35  # Lowered from 100px: allows OCR attempts on incoming two-wheelers earlier
MIN_PLATE_CHARS = 4
MAX_PLATE_CHARS = 11
OCR_MAX_WORKERS = 4

# Virtual Counting Line Coordinates (start_x_pct, start_y_pct, end_x_pct, end_y_pct)
COUNTING_LINE_PCT = (0.0, 0.65, 1.0, 0.65)

# Target Vehicle Class Mappings
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

# Pretrained COCO ID mapping
COCO_VEHICLE_ID_MAP = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Display & Runtime Settings
TARGET_MIN_FPS = 15
RTSP_RECONNECT_WAIT_SEC = 5
WINDOW_NAME = "Traffic Analysis"

# --- HELMET DETECTION CONFIGS ---
HELMET_CONF_THRESHOLD = 0.40
TWO_WHEELER_CLASSES = ["motorcycle", "scooter"]
HELMET_CHECK_EVERY_N = 3  # Check every 3 frames for faster violation convergence
HELMET_CLASS_MAP = {
    "with_helmet": "helmet",
    "without_helmet": "no_helmet",
    "helmet": "helmet",
    "no_helmet": "no_helmet",
    "head": "no_helmet",
    "Helmet": "helmet",
    "No_Helmet": "no_helmet",
    "Bike_Rider": "helmet",
}

# --- PEDESTRIAN GENDER & DEMOGRAPHIC CONFIGS ---
GENDER_CONF_THRESHOLD = 0.35
PEDESTRIAN_VEHICLE_IOU = 0.30
CHILD_HEIGHT_RATIO = 0.60
GENDER_CHECK_EVERY_N = 4    