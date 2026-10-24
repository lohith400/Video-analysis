#!/usr/bin/env python3
"""Real-time traffic video analysis — entry point."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

import cv2

# --- Ensure legacy Ultralytics namespaces are aliased before loading models ---
try:
    import ultralytics
    import ultralytics.nn
    import ultralytics.models
    sys.modules.setdefault("ultralytics.yolo", ultralytics)
    sys.modules.setdefault("ultralytics.yolo.nn", ultralytics.nn)
    sys.modules.setdefault("ultralytics.yolo.models", ultralytics.models)
except Exception:
    pass

import config
from annotator import draw_annotations
from csv_logger import CSVLogger
from detector import VehicleModelLoader, crop_vehicle, get_device
from tracker import VehicleTracker, TrackedVehicle
from traffic_counter import TrafficCounter
from helmet_checker import HelmetChecker
from pedestrian_detector import PedestrianDetector, RawBox
from snapshot_manager import SnapshotManager
from vehicle_analyzer import analyze_vehicle, make_results_dir, write_vehicle_csv


def _current_timestamp() -> str:
    return datetime.now().isoformat()


# Shared snapshot for CSV background thread
_state_lock = threading.Lock()

# Target classes initialized for cumulative reporting
_initial_target_classes = sorted(list(set(config.USER_CLASS_MAPPING.values())))
if "Others" not in _initial_target_classes:
    _initial_target_classes.append("Others")
_latest_counts: dict = {cls: 0 for cls in _initial_target_classes}
_latest_counts["total"] = 0

# Persistent session log of every pedestrian track_id ever seen
_all_pedestrians_seen: dict = {}

# Persistent session log of every vehicle track_id ever seen -> vehicle_class
_all_vehicle_classes: dict = {}


def _get_counts_snapshot() -> dict:
    with _state_lock:
        return dict(_latest_counts)


def open_capture(source: str) -> cv2.VideoCapture:
    # Convert Windows UNC path to Linux path automatically
    if source.startswith("\\\\wsl.localhost\\Ubuntu"):
        source = source.replace("\\\\wsl.localhost\\Ubuntu", "").replace("\\", "/")
    elif source.startswith("\\wsl.localhost\\Ubuntu"):
        source = source.replace("\\wsl.localhost\\Ubuntu", "").replace("\\", "/")

    cap = cv2.VideoCapture(source)
    if source.lower().startswith("rtsp://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_frame_with_reconnect(
    cap: cv2.VideoCapture, source: str
) -> tuple[cv2.VideoCapture, bool, object]:
    """Read one frame; on RTSP failure reconnect after 5s (infinite retry)."""
    is_rtsp = source.lower().startswith("rtsp://")
    while True:
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, True, frame
            if not is_rtsp:
                return cap, False, None
        except Exception:
            if not is_rtsp:
                return cap, False, None

        print(f"[RTSP] Connection lost. Reconnecting in {config.RTSP_RECONNECT_WAIT_SEC}s...")
        cap.release()
        time.sleep(config.RTSP_RECONNECT_WAIT_SEC)
        cap = open_capture(source)
        if not cap.isOpened():
            continue
        print("[RTSP] Reconnected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time traffic analysis with YOLOv8 + EasyOCR"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path or RTSP URL (e.g. rtsp://user:pass@ip:554/stream)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip plate detection and OCR — run vehicle detection only",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI video window",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = args.source
    if source.startswith("\\\\wsl.localhost\\Ubuntu"):
        source = source.replace("\\\\wsl.localhost\\Ubuntu", "").replace("\\", "/")
    elif source.startswith("\\wsl.localhost\\Ubuntu"):
        source = source.replace("\\wsl.localhost\\Ubuntu", "").replace("\\", "/")

    use_ocr = not args.no_ocr

    if not source.lower().startswith("rtsp://") and not Path(source).exists():
        print(f"Error: source not found: {source}", file=sys.stderr)
        print("Tip: Use Linux path like /home/lohit/... instead of Windows UNC path", file=sys.stderr)
        return 1

    device = get_device()
    print(f"Using device: {device}")
    print(f"OCR/Plate detection: {'enabled' if use_ocr else 'DISABLED'}")

    loader = VehicleModelLoader(device)
    tracker = VehicleTracker(loader.yolo, device, loader.vehicle_class_ids)
    
    # Initialize 2D vector geometry counting line
    counter = TrafficCounter()

    # Load OCR engine
    ocr = None
    if use_ocr:
        from ocr_engine import OCREngine
        ocr = OCREngine(device)

    helmet_checker = HelmetChecker(device=device)
    pedestrian_detector = PedestrianDetector(device=device)

    # One best snapshot per vehicle -> plate + helmet analysed once on that image
    snaps = SnapshotManager()
    results_dir = make_results_dir()
    analysis: dict = {}

    def finalize(items) -> None:
        for tid, rec in items:
            if tid not in counter.counted_ids:
                continue  # only vehicles that crossed the counting line are reported
            cls = counter.track_assigned_class.get(tid, "Others")
            if cls in ("Others", "Unknown"):
                raw = _all_vehicle_classes.get(tid, rec["raw_class"])
                cls = config.USER_CLASS_MAPPING.get(raw, "Others")
            analysis[tid] = analyze_vehicle(tid, cls, rec, ocr if use_ocr else None, helmet_checker, results_dir)
            print(f"[Snapshot] #{tid} {cls} plate={analysis[tid]['plate']} verdict={analysis[tid]['verdict']}")

    csv_logger = CSVLogger(
        counts_getter=_get_counts_snapshot,
        plates_getter=ocr.get_all_plates if use_ocr and ocr else (lambda: {}),
        violations_getter=helmet_checker.get_active_violations,
        pedestrians_getter=lambda: pedestrian_detector.get_current_pedestrians()[0],
    )
    csv_logger.start()

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source: {source}", file=sys.stderr)
        if use_ocr and ocr:
            ocr.shutdown()
        csv_logger.stop()
        return 1

    frame_idx = 0
    fps = 0.0
    t_prev = time.perf_counter()

    headless = args.headless or os.getenv("IRIS_HEADLESS", "").lower() in ("1", "true", "yes")
    if not headless:
        print("Press 'q' in the video window to quit.")
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            cap, ok, frame = read_frame_with_reconnect(cap, source)
            if not ok:
                break

            vehicles = tracker.track(frame)
            for v in vehicles:
                _all_vehicle_classes[v.track_id] = v.vehicle_class
            
            # Update 2D vector line crossing metrics
            counter.update(vehicles, frame.shape)
            
            # Sync snapshot for background CSV logger
            with _state_lock:
                global _latest_counts
                _latest_counts = counter.get_counts()

            # ── Plate detection — synchronous per active frame
            if use_ocr and ocr:
                if frame_idx % config.PLATE_DETECT_EVERY_N_FRAMES == 0:
                    plate_vehicles = [
                        v for v in vehicles
                        if v.vehicle_class in config.PLATE_DETECTION_CLASSES
                        and v.vehicle_class not in config.NO_PLATE_CLASSES
                        and (v.bbox[3] - v.bbox[1]) >= config.MIN_VEHICLE_HEIGHT_FOR_OCR
                    ]
                    ocr.detect_plates_sync(frame, plate_vehicles)

            # ── Best-snapshot capture (helmet + final plate read happen on this image)
            snaps.update(frame, vehicles, frame_idx)
            finalize(snaps.pop_lost(frame_idx))

            # ── Pedestrian & demographic detector (IRIS v2.1)
            if pedestrian_detector is not None:
                vehicle_boxes = [v.bbox for v in vehicles]
                vehicle_classes = [v.vehicle_class for v in vehicles]
                pedestrians = pedestrian_detector.detect_and_track(frame, vehicle_boxes, vehicle_classes)
                for p in pedestrians:
                    if pedestrian_detector.should_check(p.track_id, frame_idx):
                        crop = crop_vehicle(frame, p.bbox)
                        pedestrian_detector.submit(
                            p.track_id,
                            crop,
                            _current_timestamp(),
                            bbox=p.bbox,
                            frame_shape=frame.shape[:2],
                        )
                pedestrian_detector.drain_completed()

            if frame_idx % 30 == 0:
                active_ids = {v.track_id for v in vehicles}
                helmet_checker.cleanup_stale(active_ids)
                active_person_ids = {p.track_id for p in (pedestrians if pedestrian_detector else [])}
                pedestrian_detector.cleanup_stale(active_person_ids)

            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            t_prev = t_now

            plate_map = ocr.get_all_plates() if use_ocr and ocr else {}
            plate_boxes = ocr.get_all_plate_boxes() if use_ocr and ocr else {}
            total_plates = ocr.total_plates_detected if use_ocr and ocr else 0
            active_violations = helmet_checker.get_active_violations()
            pedestrian_summary, pedestrian_details = pedestrian_detector.get_current_pedestrians()
            pedestrians_data = {**pedestrian_summary, "details": pedestrian_details}

            annotated = draw_annotations(
                frame,
                vehicles,
                plate_map,
                _get_counts_snapshot(),
                fps,
                total_plates,
                counter=counter,
                plate_boxes=plate_boxes,
                violations=active_violations,
                pedestrians=pedestrians_data,
            )

            # Auto-scale display to fit screen
            h_disp, w_disp = annotated.shape[:2]
            max_w, max_h = 1280, 720
            scale = min(max_w / w_disp, max_h / h_disp, 1.0)
            if scale < 1.0:
                display_frame = cv2.resize(
                    annotated,
                    (int(w_disp * scale), int(h_disp * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                display_frame = annotated

            if not headless:
                cv2.imshow(config.WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_logger.stop()
        if use_ocr and ocr:
            ocr.shutdown()
        helmet_checker.shutdown()
        pedestrian_detector.shutdown()

    # Analyse vehicles still in view when the video ended
    finalize(snaps.pop_all())
    csv_path = write_vehicle_csv(list(analysis.values()), results_dir)
    print(f"[Snapshot] {len(analysis)} vehicle snapshots saved to {results_dir}")
    print(f"[Snapshot] Per-vehicle CSV: {csv_path}")

    # Final summary output
    final_counts = counter.get_counts()
    print("\n" + "═" * 60)
    print(" 🚗💨   INDIAN ROAD INTELLIGENCE SYSTEM - REPORT SUMMARY   🚗💨 ")
    print("═" * 60)
    print(f" {'VEHICLE TYPE':<25} ║ {'CUMULATIVE COUNT':<20}")
    print("═" * 60)
    for cls, cnt in sorted(final_counts.items()):
        if cls != "total":
            print(f" {cls:<25} ║ {cnt:<20}")
    print("─" * 60)
    print(f" {'💥 TOTAL COUNTED':<25} ║ {final_counts.get('total', 0):<20}")
    print("═" * 60)
    print(f" Logs saved to: {config.CSV_PATH}\n")

    # ── Final per-vehicle report builder
    per_vehicle_rows = []
    for track_id in sorted(counter.counted_ids):
        row = analysis.get(track_id)
        vehicle_class = (row or {}).get("vehicle_class") or counter.track_assigned_class.get(track_id, "Unknown")
        plate = (row or {}).get("plate") if row else (ocr.get_plate(track_id) if use_ocr and ocr else None)
        helmet_status = "N/A"
        if config.is_two_wheeler(vehicle_class):
            v = (row or {}).get("verdict", "")
            helmet_status = "VIOLATION (No Helmet)" if v.startswith("VIOLATION") else (
                "Compliant" if v == "COMPLIANT" else "Unverified")
        per_vehicle_rows.append({
            "track_id": track_id, "vehicle_class": vehicle_class,
            "plate": plate or "not detected", "helmet_status": helmet_status,
        })

    # ── 1. Recognized Plates table
    all_detected_plates = ocr.get_all_plates() if use_ocr and ocr else {}
    recognized_plates_rows = []
    if all_detected_plates:
        print("─" * 74)
        print(" 🎯   ALL RECOGNIZED VEHICLE LICENSE PLATES (SESSION)")
        print("─" * 74)
        print(f" {'TRACK ID':<10} ║ {'VEHICLE CLASSIFICATION':<24} ║ {'NUMBER PLATE':<16} ║ {'CONFIDENCE'}")
        print("─" * 74)
        for tid in sorted(all_detected_plates.keys()):
            plate_str = all_detected_plates[tid]
            v_class = counter.get_voted_class(tid)
            if v_class in ("Others", "Unknown") and tid in _all_vehicle_classes:
                raw_cls = _all_vehicle_classes[tid]
                v_class = config.USER_CLASS_MAPPING.get(raw_cls, raw_cls.title())
            conf = ocr.plate_confidences.get(tid, 0.0)
            conf_str = f"{conf * 100:.1f}%" if conf > 0 else "Confirmed"
            recognized_plates_rows.append({
                "track_id": tid,
                "vehicle_class": v_class,
                "plate": plate_str,
                "confidence": conf_str,
            })
            tid_str = f"#{tid}"
            print(f" {tid_str:<10} ║ {v_class:<24} ║ {plate_str:<16} ║ {conf_str}")
        print("═" * 74 + "\n")

    # ── 2. Two-wheeler helmet table: ONLY counted two-wheelers, one row per vehicle
    two_wheeler_rows = []
    tw_rows = [r for r in analysis.values() if config.is_two_wheeler(r["vehicle_class"])]
    if tw_rows:
        print("─" * 110)
        print(" 🏍️   TWO-WHEELER RIDERS & PILLIONS — HELMET COMPLIANCE REPORT (1 snapshot per vehicle)")
        print("─" * 110)
        print(f" {'TRACK ID':<10} ║ {'VEHICLE TYPE':<18} ║ {'NUMBER PLATE':<16} ║ {'RIDER':<16} ║ {'PILLION':<16} ║ {'VERDICT'}")
        print("─" * 94)
        tw_violations = tw_compliant = tw_unverified = 0
        for r in sorted(tw_rows, key=lambda x: x["track_id"]):
            v = r["verdict"]
            if v.startswith("VIOLATION"):
                tw_violations += 1
            elif v == "COMPLIANT":
                tw_compliant += 1
            else:
                tw_unverified += 1
            two_wheeler_rows.append({
                "track_id": r["track_id"], "vehicle_class": r["vehicle_class"], "plate": r["plate"],
                "rider_helmet": r["rider_helmet"], "pillion_helmet": r["pillion_helmet"], "verdict": v,
            })
            print(f" {'#' + str(r['track_id']):<10} ║ {r['vehicle_class']:<18} ║ {r['plate']:<16} ║ "
                  f"{r['rider_helmet']:<16} ║ {r['pillion_helmet']:<16} ║ {v}")
        print("─" * 94)
        print(f" 🚨 Total Two-Wheelers: {len(tw_rows)} | Violations: {tw_violations} | "
              f"Compliant: {tw_compliant} | Unverified: {tw_unverified}")
        print("═" * 94 + "\n")

    # ── 3. Crossing line vehicles
    if per_vehicle_rows:
        print("─" * 68)
        print(" 📋   ITEMIZED VEHICLE LOG (CROSSING LINE)")
        print("─" * 68)
        print(f" {'TRACK ID':<10} ║ {'VEHICLE TYPE':<18} ║ {'NUMBER PLATE':<16} ║ {'HELMET STATUS'}")
        print("─" * 68)
        for row in per_vehicle_rows:
            tid = f"#{row['track_id']}"
            v_cls = row['vehicle_class']
            plt = row['plate']
            h_stat = row['helmet_status']
            print(f" {tid:<10} ║ {v_cls:<18} ║ {plt:<16} ║ {h_stat}")
        print("═" * 68 + "\n")

    # ── 4. Pedestrian summary
    if pedestrian_detector is not None:
        pedestrian_totals = pedestrian_detector.get_session_summary()
    else:
        pedestrian_totals = {"total": 0, "males": 0, "females": 0, "children": 0, "unknown": 0}

    if pedestrian_totals and pedestrian_totals.get("total", 0) > 0:
        print("─" * 68)
        print(" 🚶   PEDESTRIAN & DEMOGRAPHIC SUMMARY")
        print("─" * 68)
        for cat, cnt in pedestrian_totals.items():
            print(f" {cat.capitalize():<25} ║ {cnt:<20}")
        print("═" * 68 + "\n")

    csv_logger.write_final_report(
        vehicle_counts=counter.get_counts(),
        per_vehicle_rows=per_vehicle_rows,
        pedestrian_totals=pedestrian_totals,
        two_wheeler_rows=two_wheeler_rows,
        recognized_plates_rows=recognized_plates_rows,
    )
    print(f" Final per-vehicle report appended to: {config.CSV_PATH}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())