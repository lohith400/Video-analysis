"""
Traffic Monitoring System - Main Pipeline
Integrates: Vehicle Detection, License Plate OCR, Helmet Detection, Gender/Age Analysis
"""

import cv2
import csv
import time
import os
from datetime import datetime
from collections import defaultdict

# ─── Import modules ───────────────────────────────────────────────────────────
from vehicle_detector import VehicleDetector
from plate_reader import PlateReader
from helmet_detector import HelmetDetector
from person_analyzer import PersonAnalyzer
from tracker import ObjectTracker
from logger import TrafficLogger

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "video_source": 0,                   # 0=webcam, or path to video file
    "vehicle_model": "yolov8n.pt",       # your existing vehicle model
    "plate_model": "license_plate_detector.pt",
    "helmet_model": "helmet_detector.pt",
    "confidence": 0.5,
    "output_dir": "output",
    "save_video": True,
    "show_display": True,
    "crossing_line_y": 0.6,              # line at 60% of frame height
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(f"{CONFIG['output_dir']}/plates", exist_ok=True)
os.makedirs(f"{CONFIG['output_dir']}/faces", exist_ok=True)


def draw_overlay(frame, stats):
    """Draw semi-transparent stats panel on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Stats panel background
    cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "TRAFFIC MONITOR", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 200), 2)

    y = 65
    for key, val in stats.items():
        cv2.putText(frame, f"{key}: {val}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25

    # Crossing line
    line_y = int(h * CONFIG["crossing_line_y"])
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 2)
    cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def process_vehicle(frame, box, track_id, vehicle_type,
                    plate_reader, logger, h):
    """Handle plate reading when vehicle crosses the line."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    line_y = int(h * CONFIG["crossing_line_y"])

    plate_text = None
    if abs(cy - line_y) < 30:           # near crossing line
        vehicle_crop = frame[y1:y2, x1:x2]
        plate_text = plate_reader.read(vehicle_crop)

        if plate_text:
            # Save plate image
            ts = datetime.now().strftime("%H%M%S%f")
            cv2.imwrite(f"{CONFIG['output_dir']}/plates/{track_id}_{ts}.jpg",
                        vehicle_crop)

        logger.log_vehicle(track_id, vehicle_type, plate_text, cx, cy)

    return plate_text


def process_person(frame, box, track_id,
                   helmet_detector, person_analyzer, logger):
    """Analyze rider/pedestrian: helmet, gender, age."""
    x1, y1, x2, y2 = box
    person_crop = frame[y1:y2, x1:x2]

    helmet_status = helmet_detector.detect(person_crop)
    gender, age, age_group = person_analyzer.analyze(person_crop)

    # Save face for record
    ts = datetime.now().strftime("%H%M%S%f")
    face_path = f"{CONFIG['output_dir']}/faces/{track_id}_{ts}.jpg"
    cv2.imwrite(face_path, person_crop)

    logger.log_person(track_id, helmet_status, gender, age, age_group)

    return helmet_status, gender, age, age_group


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(CONFIG["video_source"])
    if not cap.isOpened():
        print("❌ Could not open video source")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # ── Init modules ──────────────────────────────────────────────────────────
    print("🔧 Loading models...")
    vehicle_detector = VehicleDetector(CONFIG["vehicle_model"],
                                       CONFIG["confidence"])
    plate_reader     = PlateReader(CONFIG["plate_model"])
    helmet_detector  = HelmetDetector(CONFIG["helmet_model"])
    person_analyzer  = PersonAnalyzer()
    tracker          = ObjectTracker()
    logger           = TrafficLogger(CONFIG["output_dir"])
    print("✅ Models loaded!\n")

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if CONFIG["save_video"]:
        out_path = f"{CONFIG['output_dir']}/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(out_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # ── Stats counters ────────────────────────────────────────────────────────
    stats = defaultdict(int)
    frame_count = 0
    processed_ids = set()

    print("🎥 Processing... Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:           # skip every other frame for speed
            continue

        # ── Detect all objects ────────────────────────────────────────────────
        vehicles  = vehicle_detector.detect_vehicles(frame)
        persons   = vehicle_detector.detect_persons(frame)

        # ── Track objects ─────────────────────────────────────────────────────
        tracked_vehicles = tracker.update(vehicles, "vehicle")
        tracked_persons  = tracker.update(persons,  "person")

        # ── Process vehicles ──────────────────────────────────────────────────
        for track_id, box, vehicle_type in tracked_vehicles:
            color = (0, 200, 255)
            label = f"{vehicle_type} #{track_id}"

            if track_id not in processed_ids:
                plate = process_vehicle(frame, box, track_id, vehicle_type,
                                        plate_reader, logger, h)
                if plate:
                    label += f" | {plate}"
                    stats["Plates Read"] += 1
                processed_ids.add(track_id)
                stats[vehicle_type.title()] += 1
                stats["Total Vehicles"] += 1

            draw_box(frame, box, label, color)

        # ── Process persons ───────────────────────────────────────────────────
        for track_id, box, _ in tracked_persons:
            if track_id not in processed_ids:
                helmet, gender, age, age_group = process_person(
                    frame, box, track_id,
                    helmet_detector, person_analyzer, logger)

                # Update stats
                if helmet == "no_helmet":
                    stats["No Helmet"] += 1
                if age_group == "Child":
                    stats["Children"] += 1
                stats[gender] = stats.get(gender, 0) + 1
                processed_ids.add(track_id)

            # Color-code by helmet
            helmet_status = helmet_detector.detect(
                frame[box[1]:box[3], box[0]:box[2]])
            color = (0, 0, 255) if helmet_status == "no_helmet" else (0, 255, 0)
            draw_box(frame, box,
                     f"Person #{track_id} | {helmet_status}", color)

        # ── Draw overlay ───────────────────────────────────────────────────────
        frame = draw_overlay(frame, {
            "Vehicles":     stats["Total Vehicles"],
            "Cars":         stats.get("car", 0),
            "Bikes":        stats.get("motorcycle", 0),
            "No Helmet":    stats["No Helmet"],
            "Children":     stats["Children"],
            "Plates Read":  stats["Plates Read"],
            "FPS":          f"{fps:.0f}",
        })

        if writer:
            writer.write(frame)

        if CONFIG["show_display"]:
            cv2.imshow("Traffic Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.close()
    print("\n✅ Processing complete!")
    print(f"📁 Results saved to: {CONFIG['output_dir']}/")
    print(f"📊 Final stats: {dict(stats)}")


if __name__ == "__main__":
    main()
