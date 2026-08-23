import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
VIDEO_PATH = "/home/lohit/realcode/ML/Code/projectsss/vehicle/videos/Video Project 3.mp4" 
VEHICLE_MODEL = "yolov8n.pt"  # Lightweight and fast
PLATE_MODEL = "license_plate_detector.pt"

# Only count these COCO classes
# 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [2, 3, 5, 7]

# Counting Line Position (0.6 means 60% down the screen)
LINE_POSITION = 0.6 

# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════
reader = easyocr.Reader(['en'], gpu=True)

def clean_plate(text):
    """Basic filter for Indian Plate patterns"""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Typical Indian plate is 7-10 chars: KA 05 MG 1234
    if 7 <= len(text) <= 10:
        return text
    return None

# ═══════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════
def main():
    model = YOLO(VEHICLE_MODEL)
    plate_model = YOLO(PLATE_MODEL)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * LINE_POSITION)

    # State tracking
    track_history = {} # ID -> last centroid
    counted_ids = set()
    vehicle_counts = {"Car": 0, "Bike": 0, "Bus/Truck": 0}
    plate_registry = {} # ID -> Best Plate Found

    print("[INFO] Processing video with Line-Crossing logic...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 1. TRACKING: Use built-in ByteTrack for stability
        results = model.track(frame, persist=True, classes=TARGET_CLASSES, tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Check for Line Crossing
                if track_id in track_history:
                    prev_cy = track_history[track_id][1]
                    
                    # If it moves from above the line to below the line
                    if prev_cy < line_y <= cy and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        label = "Car" if cls == 2 else "Bike" if cls == 3 else "Bus/Truck"
                        vehicle_counts[label] += 1
                        
                        # 2. PLATE DETECTION: Only run when crossing (efficiency!)
                        roi = frame[y1:y2, x1:x2]
                        plate_results = plate_model(roi, verbose=False)
                        
                        for p_box in plate_results[0].boxes.xyxy:
                            px1, py1, px2, py2 = map(int, p_box)
                            plate_crop = roi[py1:py2, px1:px2]
                            
                            # OCR
                            ocr_res = reader.readtext(plate_crop)
                            for (_, text, conf) in ocr_res:
                                clean = clean_plate(text)
                                if clean and conf > 0.4:
                                    plate_registry[track_id] = clean

                track_history[track_id] = (cx, cy)

                # Draw Visuals
                color = (0, 255, 0) if track_id in counted_ids else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                p_text = plate_registry.get(track_id, "")
                cv2.putText(frame, f"ID:{track_id} {p_text}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw Counting Line
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 255), 3)
        
        # Display Counts
        y_offset = 50
        for k, v in vehicle_counts.items():
            cv2.putText(frame, f"{k}: {v}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 40

        cv2.imshow("Optimized Indian Traffic Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Final Totals ---")
    print(vehicle_counts)

if __name__ == "__main__":
    main()