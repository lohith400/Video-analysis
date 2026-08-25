#!/usr/bin/env python3
"""Real-time traffic video analysis — entry point."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

import config
from annotator import draw_annotations
from csv_logger import CSVLogger
from detector import VehicleModelLoader, crop_vehicle, get_device
from tracker import VehicleTracker, TrackedVehicle
from traffic_counter import TrafficCounter

# Shared snapshot for CSV background thread
_state_lock = threading.Lock()

# Target classes initialized for cumulative reporting
_initial_target_classes = sorted(list(set(config.USER_CLASS_MAPPING.values())))
if "Others" not in _initial_target_classes:
    _initial_target_classes.append("Others")
_latest_counts: dict = {cls: 0 for cls in _initial_target_classes}
_latest_counts["total"] = 0


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Normalize Windows UNC path to Linux path
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
    
    # Initialize high-precision line-crossing counter
    counter = TrafficCounter()

    # Only load OCR engine if needed
    ocr = None
    if use_ocr:
        from ocr_engine import OCREngine
        ocr = OCREngine(device)

    csv_logger = CSVLogger(
        counts_getter=_get_counts_snapshot,
        plates_getter=ocr.get_all_plates if use_ocr else (lambda: {}),
    )
    csv_logger.start()

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source: {source}", file=sys.stderr)
        if use_ocr:
            ocr.shutdown()
        csv_logger.stop()
        return 1

    frame_idx = 0
    fps = 0.0
    t_prev = time.perf_counter()

    print("Press 'q' in the video window to quit.")
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            cap, ok, frame = read_frame_with_reconnect(cap, source)
            if not ok:
                break

            vehicles = tracker.track(frame)
            
            # Update TrafficCounter state & calculate cumulative crossings
            counter.update(vehicles, frame.shape)
            
            # Sync cumulative counts snapshot for the background CSV logger
            with _state_lock:
                global _latest_counts
                _latest_counts = counter.get_counts()

            # Only run plate pipeline if OCR is enabled
            if use_ocr:
                run_plate = frame_idx % config.PLATE_DETECT_EVERY_N_FRAMES == 0
                if run_plate:
                    for v in vehicles:
                        if _should_run_plate_pipeline(v, ocr):
                            crop = crop_vehicle(frame, v.bbox)
                            if crop.size > 0:
                                ocr.submit_vehicle_crop(v.track_id, crop, v.bbox)
                ocr.drain_completed()

            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            t_prev = t_now

            plate_map = ocr.get_all_plates() if use_ocr else {}
            plate_boxes = ocr.get_all_plate_boxes() if use_ocr else {}
            total_plates = ocr.total_plates_detected if use_ocr else 0

            annotated = draw_annotations(
                frame,
                vehicles,
                plate_map,
                _get_counts_snapshot(),
                fps,
                total_plates,
                counter=counter,
                plate_boxes=plate_boxes,
            )

            # Auto-scale display to fit screen comfortably
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

    # Output a gorgeous, highly-accurate final statistics summary
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

    return 0


def _should_run_plate_pipeline(vehicle: TrackedVehicle, ocr) -> bool:
    if vehicle.vehicle_class in config.NO_PLATE_CLASSES:
        return False
    if vehicle.vehicle_class not in config.PLATE_DETECTION_CLASSES:
        return False
    
    # Check if the vehicle is large/close enough to have a readable license plate
    x1, y1, x2, y2 = vehicle.bbox
    bbox_height = y2 - y1
    if bbox_height < config.MIN_VEHICLE_HEIGHT_FOR_OCR:
        return False
        
    return ocr.needs_ocr(vehicle.track_id)


if __name__ == "__main__":
    sys.exit(main())