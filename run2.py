import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import re

# =========================
# CONFIG
# =========================
VIDEO_IN = "/home/lohit/realcode/ML/Code/projectsss/vehicle/videos/L3.mp4"

VEHICLE_MODEL = "yolov8n.pt"
PLATE_MODEL   = "license_plate_detector.pt"

ALLOWED = ["car", "motorcycle", "bus", "truck", "bicycle"]

# =========================
# LOAD MODELS
# =========================
print("Loading models...")
vehicle_model = YOLO(VEHICLE_MODEL)
plate_model   = YOLO(PLATE_MODEL)
reader        = easyocr.Reader(['en'], gpu=True)
print("Models loaded!\n")

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_IN)

if not cap.isOpened():
    print("Error opening video")
    exit()

# =========================
# HELPERS
# =========================
def is_valid_plate(text):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}$'
    return re.match(pattern, text)

def enhance_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # resize (VERY IMPORTANT)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    # slight blur
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    return gray

def safe_image(img):
    if img is None or img.size == 0:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# =========================
# OCR FUNCTION
# =========================
def read_plate(crop):

    if crop is None or crop.size == 0:
        return None, None

    results = plate_model(crop, conf=0.4, imgsz=320, verbose=False)[0]

    if not results.boxes:
        return None, None

    best = max(results.boxes, key=lambda b: float(b.conf[0]))
    px1, py1, px2, py2 = map(int, best.xyxy[0])

    px1 = max(0, px1)
    py1 = max(0, py1)
    px2 = min(crop.shape[1], px2)
    py2 = min(crop.shape[0], py2)

    plate_img = crop[py1:py2, px1:px2]

    if plate_img.size == 0:
        return None, None

    # preprocess
    enhanced = enhance_plate(plate_img)
    enhanced = safe_image(enhanced)

    if enhanced is None:
        return None, None

    # OCR
    result = reader.readtext(enhanced)

    if not result:
        return None, (px1, py1, px2, py2)

    # take best confidence result
    best = max(result, key=lambda x: x[2])

    if best[2] < 0.5:
        return None, (px1, py1, px2, py2)

    text = best[1].upper().replace(" ", "")

    # filter characters
    valid = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = "".join([c for c in text if c in valid])

    # validate Indian plate
    if not is_valid_plate(text):
        return None, (px1, py1, px2, py2)

    print("Plate:", text)

    return text, (px1, py1, px2, py2)

# =========================
# MAIN LOOP
# =========================
print("Running... Press Q to exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device=0,
        conf=0.4,
        imgsz=640,
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes     = results[0].boxes.xyxy.cpu()
        class_ids = results[0].boxes.cls.cpu()
        track_ids = results[0].boxes.id.cpu()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):

            class_name = vehicle_model.names[int(class_id)]

            if class_name not in ALLOWED:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            plate_text, plate_box = read_plate(crop)

            label = f"{class_name} ID:{int(track_id)}"
            if plate_text:
                label += f" | {plate_text}"

            # draw vehicle
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # draw plate
            if plate_box:
                fpx1 = x1 + plate_box[0]
                fpy1 = y1 + plate_box[1]
                fpx2 = x1 + plate_box[2]
                fpy2 = y1 + plate_box[3]

                cv2.rectangle(frame, (fpx1,fpy1),
                              (fpx2,fpy2), (0,0,255), 2)

    cv2.imshow("Vehicle + Plate System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

