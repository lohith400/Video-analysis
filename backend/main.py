#!/usr/bin/env python3
"""Real-time traffic video analysis — entry point."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

from datetime import datetime

import config
from annotator import draw_annotations
from csv_logger import CSVLogger
from detector import VehicleModelLoader, crop_vehicle, get_device
from tracker import VehicleTracker, TrackedVehicle
from traffic_counter import TrafficCounter
from helmet_checker import HelmetChecker
from pedestrian_detector import PedestrianDetector, RawBox


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

# Persistent, never-purged session log of every pedestrian track_id ever
# seen, with the last-known gender classification. pedestrian_detector's own
# `.results` dict gets pruned as people leave frame (cleanup_stale), so it
# can't answer "how many distinct people were seen in the whole video" on
# its own — this dict is what makes the final report's human count correct.
_all_pedestrians_seen: dict = {}


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

    # Helmet detection and pedestrian/gender detection — these already exist
    # and work (server.py has used them all along), they just were never
    # wired into this script before, which is why every CSV produced by
    # main.py showed helmet_violations=0 and pedestrians=0 regardless of
    # what was actually in the video.
    helmet_checker = HelmetChecker(device=device)
    pedestrian_detector = PedestrianDetector(device=device)

    csv_logger = CSVLogger(
        counts_getter=_get_counts_snapshot,
        plates_getter=ocr.get_all_plates if use_ocr else (lambda: {}),
        violations_getter=helmet_checker.get_active_violations,
        pedestrians_getter=lambda: pedestrian_detector.get_current_pedestrians()[0],
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

            # ── Plate detection — synchronous, every frame (see detect_plates_sync
            # docstring in ocr_engine.py for why this replaced the old buffered
            # background-thread approach: a moving camera makes any stale/buffered
            # detection land in the wrong place, both visually and for what the
            # OCR crop actually contains).
            if use_ocr:
                if frame_idx % config.PLATE_DETECT_EVERY_N_FRAMES == 0:
                    plate_vehicles = [
                        v for v in vehicles
                        if v.vehicle_class in config.PLATE_DETECTION_CLASSES
                        and v.vehicle_class not in config.NO_PLATE_CLASSES
                        and (v.bbox[3] - v.bbox[1]) >= config.MIN_VEHICLE_HEIGHT_FOR_OCR
                    ]
                    ocr.detect_plates_sync(frame, plate_vehicles)

            # ── Helmet checker (async — bbox precision doesn't matter here,
            # only the classification, so the existing background thread pool
            # design from server.py is fine as-is).
            if helmet_checker is not None:
                for v in vehicles:
                    if v.vehicle_class in config.TWO_WHEELER_CLASSES:
                        with helmet_checker._lock:
                            if v.track_id not in helmet_checker.all_two_wheeler_statuses:
                                plate_now = ocr.get_plate(v.track_id) if use_ocr else None
                                helmet_checker.all_two_wheeler_statuses[v.track_id] = {
                                    "track_id": v.track_id,
                                    "plate": plate_now or "UNKNOWN",
                                    "vehicle_class": v.vehicle_class,
                                    "rider_helmet": "unknown",
                                    "pillion_helmet": "none",
                                    "timestamp": _current_timestamp(),
                                }
                        if helmet_checker.should_check(v.track_id, frame_idx):
                            plate = (ocr.get_plate(v.track_id) if use_ocr else None) or "UNKNOWN"
                            crop = crop_vehicle(frame, v.bbox)
                            helmet_checker.submit(v.track_id, crop, v.vehicle_class, plate, _current_timestamp(), v.bbox)
                helmet_checker.drain_completed()

            # ── Pedestrian & gender/child detector.
            if pedestrian_detector is not None:
                raw_results = []
                if tracker.last_boxes is not None:
                    boxes = tracker.last_boxes
                    if boxes.id is not None:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        for i in range(len(xyxy)):
                            raw_results.append(RawBox(
                                cls=cls_ids[i],
                                track_id=track_ids[i],
                                bbox=(int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])),
                            ))
                person_boxes = [b for b in raw_results if b.cls == 0]
                vehicle_boxes = [v.bbox for v in vehicles]
                pedestrians = pedestrian_detector.filter_pedestrians(person_boxes, vehicle_boxes)
                for p in pedestrians:
                    if pedestrian_detector.should_check(p.track_id, frame_idx):
                        crop = crop_vehicle(frame, p.bbox)
                        pedestrian_detector.submit(p.track_id, crop, _current_timestamp())
                pedestrian_detector.drain_completed()

                # Record into the persistent session log BEFORE cleanup_stale
                # prunes anything — this is what lets the final report count
                # every distinct person seen across the whole video, not just
                # whoever happens to still be on-screen at the last frame.
                for p in pedestrians:
                    gender = pedestrian_detector.results.get(p.track_id, "unknown")
                    if p.track_id not in _all_pedestrians_seen or gender != "unknown":
                        _all_pedestrians_seen[p.track_id] = gender

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

            plate_map = ocr.get_all_plates() if use_ocr else {}
            plate_boxes = ocr.get_all_plate_boxes() if use_ocr else {}
            total_plates = ocr.total_plates_detected if use_ocr else 0
            active_violations = helmet_checker.get_active_violations()
            pedestrian_summary, _ = pedestrian_detector.get_current_pedestrians()

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
                pedestrians=pedestrian_summary,
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
        helmet_checker.shutdown()
        pedestrian_detector.shutdown()

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

    # ── Final per-vehicle report, appended to the SAME csv file ────────────
    # Built from counter.counted_ids / counter.track_assigned_class, which
    # (unlike the live plate/helmet dicts) are never purged mid-run, so this
    # covers every vehicle actually counted across the whole video, not just
    # whichever tracks happened to still be on screen at the final frame.
    per_vehicle_rows = []
    for track_id in sorted(counter.counted_ids):
        vehicle_class = counter.track_assigned_class.get(track_id, "Unknown")
        plate = ocr.get_plate(track_id) if use_ocr and ocr else None
        helmet_status = "N/A"
        if vehicle_class in config.TWO_WHEELER_CLASSES:
            status = helmet_checker.all_two_wheeler_statuses.get(track_id, {})
            helmet_status = status.get("rider_helmet", "unknown")
        per_vehicle_rows.append({
            "track_id": track_id,
            "vehicle_class": vehicle_class,
            "plate": plate or "not detected",
            "helmet_status": helmet_status,
        })

    pedestrian_totals = {"total": 0, "males": 0, "females": 0, "children": 0, "unknown": 0}
    for gender in _all_pedestrians_seen.values():
        pedestrian_totals["total"] += 1
        if gender == "male_adult":
            pedestrian_totals["males"] += 1
        elif gender == "female_adult":
            pedestrian_totals["females"] += 1
        elif gender == "child":
            pedestrian_totals["children"] += 1
        else:
            pedestrian_totals["unknown"] += 1

    csv_logger.write_final_report(
        vehicle_counts=counter.get_counts(),
        per_vehicle_rows=per_vehicle_rows,
        pedestrian_totals=pedestrian_totals,
    )
    print(f" Final per-vehicle report appended to: {config.CSV_PATH}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())