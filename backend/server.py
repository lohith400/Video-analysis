#!/usr/bin/env python3
"""
FastAPI backend for Indian Road Intelligence System.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

If license_plate_detector.pt is missing, plate detection is skipped
automatically — the video feed will still stream normally.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from datetime import datetime

def current_timestamp() -> str:
    return datetime.now().isoformat()

try:
    from fastapi import Depends, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Run: pip install fastapi uvicorn python-multipart")

import os
import secrets

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads backend/.env if present; no-op otherwise
except ImportError:
    pass  # python-dotenv is optional — env vars can be set another way

import config
from annotator import draw_annotations
from auth import API_KEY, require_api_key
from detector import VehicleModelLoader, crop_vehicle, get_device, crop_vehicle as crop_person
from tracker import VehicleTracker
from helmet_checker import HelmetChecker
from pedestrian_detector import PedestrianDetector, RawBox

# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="Indian Road Intelligence System")

# CORS origins come from env (comma-separated) so a deployed frontend isn't
# stuck behind a wildcard. Defaults to permissive for local dev only.
_cors_origins_env = os.getenv("IRIS_CORS_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not API_KEY:
    print(
        "[server] WARNING: IRIS_API_KEY is not set — all REST/WebSocket "
        "endpoints are UNAUTHENTICATED. Set IRIS_API_KEY before exposing "
        "this server beyond localhost."
    )

# ── Runtime state ──────────────────────────────────────────────────────────
_state_lock = threading.Lock()

# Target classes initialized for cumulative reporting
_target_classes = sorted(list(set(config.USER_CLASS_MAPPING.values())))
if "Others" not in _target_classes:
    _target_classes.append("Others")
_latest_counts:     Dict           = {cls: 0 for cls in _target_classes}
_latest_counts["total"] = 0
_latest_plates:     Dict[int, str] = {}
_latest_violations: List[Dict]     = []
_latest_two_wheeler_statuses: List[Dict] = []
_latest_pedestrians: Dict          = {"total": 0, "males": 0, "females": 0, "children": 0, "details": []}
_latest_fps:        float          = 0.0
_source_type:       str            = "VIDEO"
_is_running:        bool           = False

# Overall-video summary (populated when a file finishes processing)
_video_done:    bool           = False
_video_summary: Optional[Dict] = None   # {"counts": {...}, "plates": [...]}

# Frames pushed from analysis thread → WebSocket (newest wins)
_frame_queue: "queue.Queue[str]" = queue.Queue(maxsize=2)

# ── Pipeline singletons ────────────────────────────────────────────────────
_device  = get_device()
_loader:  Optional[VehicleModelLoader] = None
_tracker: Optional[VehicleTracker]     = None
_ocr                                   = None
_helmet_checker: Optional[HelmetChecker] = None
_pedestrian_detector: Optional[PedestrianDetector] = None
_pipeline_lock = threading.Lock()

_analysis_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_pipeline() -> None:
    global _loader, _tracker, _ocr, _helmet_checker, _pedestrian_detector
    with _pipeline_lock:
        if _loader is not None:
            return
        print(f"[server] Loading vehicle model on {_device} …")
        _loader  = VehicleModelLoader(_device)
        _tracker = VehicleTracker(_loader.yolo, _device, _loader.vehicle_class_ids)
        print("[server] Vehicle model ready ✓")
        try:
            from ocr_engine import OCREngine
            _ocr = OCREngine(_device)
            print("[server] OCR engine ready ✓")
        except FileNotFoundError as exc:
            print(f"[server] WARNING: {exc} — plate detection disabled.")
            _ocr = None
        except Exception as exc:
            print(f"[server] WARNING: OCR init failed ({exc}) — plate detection disabled.")
            _ocr = None

        # Initialize both at server startup alongside existing pipeline:
        try:
            _helmet_checker = HelmetChecker(device=_device)
            print("[server] Helmet checker ready ✓")
        except Exception as exc:
            print(f"[server] WARNING: HelmetChecker init failed ({exc}) — helmet detection disabled.")
            _helmet_checker = None

        try:
            _pedestrian_detector = PedestrianDetector(device=_device)
            print("[server] Pedestrian detector ready ✓")
        except Exception as exc:
            print(f"[server] WARNING: PedestrianDetector init failed ({exc}) — pedestrian detection disabled.")
            _pedestrian_detector = None


def _frame_to_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def _push_frame(b64: str) -> None:
    if not b64:
        return
    if _frame_queue.full():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        _frame_queue.put_nowait(b64)
    except queue.Full:
        pass


def _flush_queue() -> None:
    while not _frame_queue.empty():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            break


def _stop_current(timeout: float = 3.0) -> None:
    global _analysis_thread
    _stop_event.set()
    if _analysis_thread and _analysis_thread.is_alive():
        _analysis_thread.join(timeout=timeout)
    _analysis_thread = None


def _reset_summary() -> None:
    """Clear previous video summary before starting a new source."""
    global _video_done, _video_summary, _latest_violations, _latest_pedestrians, _latest_two_wheeler_statuses, _latest_plates
    with _state_lock:
        _video_done    = False
        _video_summary = None
        _latest_violations = []
        _latest_two_wheeler_statuses = []
        _latest_plates = {}
        _latest_pedestrians = {"total": 0, "males": 0, "females": 0, "children": 0, "details": []}
    # Also clear the OCR engine's accumulated results so plates from a previous
    # video session don't bleed into the new one
    if _ocr is not None:
        with _ocr._lock:
            _ocr.results.clear()
            _ocr.plate_boxes.clear()
            _ocr.pending_futures.clear()
            _ocr.attempts.clear()
            _ocr.vehicle_crop_buffer.clear()
            _ocr.ocr_history.clear()
    # Clear helmet checker session state
    if _helmet_checker is not None:
        with _helmet_checker._lock:
            _helmet_checker.active_violations.clear()
            _helmet_checker.two_wheeler_status.clear()
            _helmet_checker.all_two_wheeler_statuses.clear()
            _helmet_checker.attempts.clear()
            _helmet_checker.pending_futures.clear()


# ── Analysis thread ────────────────────────────────────────────────────────

def _run_analysis(source, source_type: str) -> None:
    global _latest_counts, _latest_fps, _source_type, _is_running
    global _video_done, _video_summary

    try:
        _load_pipeline()
    except Exception as exc:
        print(f"[server] FATAL: Could not load vehicle model: {exc}")
        return

    _source_type = source_type
    _stop_event.clear()
    _flush_queue()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[server] ERROR: Cannot open source: {source!r}")
        return

    is_live_source = source_type in ("LIVE", "WEBCAM")
    is_rtsp        = isinstance(source, str) and source.lower().startswith("rtsp://")
    if is_rtsp:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with _state_lock:
        _is_running = True

    csv_logger = None
    try:
        from csv_logger import CSVLogger
        csv_logger = CSVLogger(
            counts_getter=lambda: dict(_latest_counts),
            plates_getter=lambda: (dict(_latest_plates) if _ocr else {}),
            violations_getter=lambda: list(_latest_violations),
            pedestrians_getter=lambda: dict(_latest_pedestrians),
        )
        csv_logger.start()
    except Exception as exc:
        print(f"[server] CSV logger disabled: {exc}")

    fps       = 0.0
    t_prev    = time.perf_counter()
    frame_idx = 0

    # Initialize high-precision line-crossing counter
    from traffic_counter import TrafficCounter
    counter = TrafficCounter()

    print(f"[server] ▶ Analysis started — source={source!r}  type={source_type}")

    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                if is_rtsp:
                    print("[server] RTSP lost — reconnecting…")
                    cap.release()
                    time.sleep(config.RTSP_RECONNECT_WAIT_SEC)
                    cap = cv2.VideoCapture(source)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                else:
                    print("[server] ■ End of video file.")
                    break

            # ── Vehicle tracking & Cumulative Line-Crossing ─────────────────
            try:
                vehicles = _tracker.track(frame)
                counter.update(vehicles, frame.shape)
            except Exception as exc:
                print(f"[server] Tracker error (frame {frame_idx}): {exc}")
                vehicles = []

            # Retrieve cumulative statistics
            counts = counter.get_counts()

            # ── Plate OCR ───────────────────────────────────────────────────
            plates_map: Dict[int, str] = {}
            plate_boxes_map: Dict[int, tuple] = {}
            if _ocr is not None:
                if frame_idx % config.PLATE_DETECT_EVERY_N_FRAMES == 0:
                    for v in vehicles:
                        # Only run OCR if the vehicle crop is large/close enough
                        x1, y1, x2, y2 = v.bbox
                        bbox_height = y2 - y1
                        if bbox_height < config.MIN_VEHICLE_HEIGHT_FOR_OCR:
                            continue

                        if (
                            v.vehicle_class in config.PLATE_DETECTION_CLASSES
                            and v.vehicle_class not in config.NO_PLATE_CLASSES
                            and _ocr.needs_ocr(v.track_id)
                        ):
                            try:
                                crop = crop_vehicle(frame, v.bbox)
                                if crop.size > 0:
                                    _ocr.submit_vehicle_crop(v.track_id, crop, v.bbox)
                            except Exception:
                                pass
                try:
                    _ocr.drain_completed()
                    plates_map = _ocr.get_all_plates()
                    plate_boxes_map = _ocr.get_all_plate_boxes()
                except Exception:
                    pass

            # ── Helmet Checker (Module 1) ───────────────────────────────────
            all_violations = []
            two_wheeler_statuses = []
            if _helmet_checker is not None:
                try:
                    for v in vehicles:
                        if v.vehicle_class in config.TWO_WHEELER_CLASSES:
                            # ── INSTANT LOG: add every two-wheeler immediately on first detection
                            # so it appears in the table right away (status = unknown until model runs)
                            with _helmet_checker._lock:
                                if v.track_id not in _helmet_checker.all_two_wheeler_statuses:
                                    plate_now = "UNKNOWN"
                                    if _ocr is not None:
                                        plate_now = _ocr.get_plate(v.track_id) or "UNKNOWN"
                                    _helmet_checker.all_two_wheeler_statuses[v.track_id] = {
                                        "track_id": v.track_id,
                                        "plate": plate_now,
                                        "vehicle_class": v.vehicle_class,
                                        "rider_helmet": "unknown",
                                        "pillion_helmet": "none",
                                        "timestamp": current_timestamp()
                                    }

                            if _helmet_checker.should_check(v.track_id, frame_idx):
                                plate = "UNKNOWN"
                                if _ocr is not None:
                                    plate = _ocr.get_plate(v.track_id) or "UNKNOWN"
                                crop = crop_vehicle(frame, v.bbox)
                                _helmet_checker.submit(v.track_id, crop, v.vehicle_class, plate, current_timestamp(), v.bbox)

                    _helmet_checker.drain_completed()
                    all_violations = _helmet_checker.get_active_violations()
                    two_wheeler_statuses = _helmet_checker.get_two_wheeler_statuses()
                except Exception as exc:
                    print(f"[server] Helmet checker error: {exc}")

            # ── Pedestrian & Gender/Child Detector (Module 2) ───────────────
            pedestrians = []
            pedestrian_summary = {"total": 0, "males": 0, "females": 0, "children": 0}
            pedestrian_details = []
            
            if _pedestrian_detector is not None:
                try:
                    raw_results = []
                    if _tracker.last_boxes is not None:
                        boxes = _tracker.last_boxes
                        if boxes.id is not None:
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls_ids = boxes.cls.cpu().numpy().astype(int)
                            track_ids = boxes.id.cpu().numpy().astype(int)
                            for i in range(len(xyxy)):
                                raw_results.append(RawBox(
                                    cls=cls_ids[i],
                                    track_id=track_ids[i],
                                    bbox=(int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]))
                                ))
                                
                    person_boxes = [b for b in raw_results if b.cls == 0]
                    vehicle_boxes = [v.bbox for v in vehicles]
                    pedestrians = _pedestrian_detector.filter_pedestrians(person_boxes, vehicle_boxes)
                    
                    for p in pedestrians:
                        if _pedestrian_detector.should_check(p.track_id, frame_idx):
                            crop = crop_person(frame, p.bbox)
                            _pedestrian_detector.submit(p.track_id, crop, current_timestamp())
                            
                    _pedestrian_detector.drain_completed()
                    pedestrian_summary, pedestrian_details = _pedestrian_detector.get_current_pedestrians()
                except Exception as exc:
                    print(f"[server] Pedestrian detector error: {exc}")

            # ── Stale Teardown Cleanup ──────────────────────────────────────
            if frame_idx % 30 == 0:
                if _helmet_checker is not None:
                    active_ids = {v.track_id for v in vehicles}
                    _helmet_checker.cleanup_stale(active_ids)
                if _pedestrian_detector is not None:
                    active_person_ids = {p.track_id for p in pedestrians}
                    _pedestrian_detector.cleanup_stale(active_person_ids)

            # ── FPS ─────────────────────────────────────────────────────────
            t_now = time.perf_counter()
            dt    = t_now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt
            t_prev = t_now

            # ── Shared state update ─────────────────────────────────────────
            with _state_lock:
                _latest_counts = counts
                _latest_plates = plates_map
                _latest_violations = all_violations
                _latest_two_wheeler_statuses = two_wheeler_statuses
                _latest_pedestrians = {
                    "total": pedestrian_summary.get("total", 0),
                    "males": pedestrian_summary.get("males", 0),
                    "females": pedestrian_summary.get("females", 0),
                    "children": pedestrian_summary.get("children", 0),
                    "details": pedestrian_details
                }
                _latest_fps    = fps

            # ── Annotate + push frame ───────────────────────────────────────
            try:
                total_plates = _ocr.total_plates_detected if _ocr else 0
                annotated    = draw_annotations(
                    frame,
                    vehicles,
                    plates_map,
                    counts,
                    fps,
                    total_plates,
                    counter=counter,
                    plate_boxes=plate_boxes_map,
                    violations=all_violations,
                    pedestrians={
                        "total": pedestrian_summary.get("total", 0),
                        "males": pedestrian_summary.get("males", 0),
                        "females": pedestrian_summary.get("females", 0),
                        "children": pedestrian_summary.get("children", 0),
                        "details": pedestrian_details
                    }
                )
            except Exception as exc:
                print(f"[server] Annotation error: {exc}")
                annotated = frame

            _push_frame(_frame_to_b64(annotated))
            frame_idx += 1

    except Exception as exc:
        print(f"[server] Unexpected analysis error: {exc}")

    finally:
        cap.release()
        if csv_logger:
            try:
                csv_logger.stop()
            except Exception:
                pass

        if _helmet_checker is not None:
            try:
                _helmet_checker.shutdown()
            except Exception:
                pass

        if _pedestrian_detector is not None:
            try:
                _pedestrian_detector.shutdown()
            except Exception:
                pass

        # ── Build overall-video summary when a FILE finishes naturally ──────
        # (not for live/RTSP, and not when manually stopped)
        if not is_live_source and not _stop_event.is_set():
            # Use exact high-precision line crossing count for identical final summary
            final_counts = counter.get_counts()

            final_plates: List[Dict] = []
            if _ocr:
                try:
                    final_plates = [
                        {"plate": p, "timestamp": int(time.time() * 1000)}
                        for p in _ocr.get_all_plates().values()
                    ]
                except Exception:
                    pass

            final_violations = []
            final_two_wheeler_statuses = []
            if _helmet_checker is not None:
                final_violations = _helmet_checker.all_violations
                final_two_wheeler_statuses = _helmet_checker.get_two_wheeler_statuses()

            final_pedestrians = {"total": 0, "males": 0, "females": 0, "children": 0}
            if _pedestrian_detector is not None:
                try:
                    ped_sum, _ = _pedestrian_detector.get_current_pedestrians()
                    final_pedestrians = {
                        "total": ped_sum.get("total", 0),
                        "males": ped_sum.get("males", 0),
                        "females": ped_sum.get("females", 0),
                        "children": ped_sum.get("children", 0)
                    }
                except Exception:
                    pass

            with _state_lock:
                _video_done    = True
                _video_summary = {
                    "counts": final_counts,
                    "plates": final_plates,
                    "violations": final_violations,
                    "two_wheeler_statuses": final_two_wheeler_statuses,
                    "pedestrians": final_pedestrians
                }

            print(
                f"[server] ✓ Video complete — "
                f"{final_counts['total']} cumulative vehicles, "
                f"{len(final_plates)} plates."
            )

        with _state_lock:
            _is_running = False

        print("[server] ■ Analysis thread exited.")


# ── REST endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    with _state_lock:
        running = _is_running
    return {
        "status":           "ok",
        "device":           _device,
        "models_loaded":    _loader is not None,
        "ocr_available":    _ocr is not None,
        "analysis_running": running,
    }


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload(file: UploadFile = File(...)):
    global _analysis_thread

    dest_dir = Path("uploads")
    dest_dir.mkdir(exist_ok=True)
    safe_name = Path(file.filename).name
    dest = dest_dir / safe_name
    dest.write_bytes(await file.read())
    print(f"[server] Saved upload → {dest}  ({dest.stat().st_size:,} bytes)")

    mime        = (file.content_type or "").lower()
    source_type = "VIDEO" if "video" in mime else "IMAGE"

    _stop_current()
    _reset_summary()   # clear old summary before new file

    _analysis_thread = threading.Thread(
        target=_run_analysis,
        args=(str(dest), source_type),
        daemon=True,
        name="analysis",
    )
    _analysis_thread.start()

    return {"status": "started", "file": safe_name, "source_type": source_type}


class ConnectRequest(BaseModel):
    source: str


@app.post("/connect", dependencies=[Depends(require_api_key)])
async def connect(req: ConnectRequest):
    global _analysis_thread

    raw = req.source.strip()
    if raw.lower() == "webcam":
        source, source_type = 0, "WEBCAM"
    else:
        source, source_type = raw, "LIVE"

    _stop_current()
    _reset_summary()   # live streams never produce a video summary

    _analysis_thread = threading.Thread(
        target=_run_analysis,
        args=(source, source_type),
        daemon=True,
        name="analysis",
    )
    _analysis_thread.start()

    return {"status": "connecting", "source": req.source}


@app.post("/stop", dependencies=[Depends(require_api_key)])
async def stop():
    _stop_current()
    _reset_summary()
    return {"status": "stopped"}


@app.get("/analytics")
async def analytics():
    csv_path = Path(config.CSV_PATH)
    if not csv_path.exists():
        return {"rows": []}
    rows: List[Dict] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    except Exception as exc:
        print(f"[server] CSV read error: {exc}")
    return {"rows": rows}


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/video-feed")
async def video_feed(websocket: WebSocket):
    # WebSocket handshakes can't carry custom headers from a browser, so the
    # key is passed as a query param instead: /video-feed?api_key=...
    if API_KEY:
        supplied = websocket.query_params.get("api_key", "")
        if not secrets.compare_digest(supplied, API_KEY):
            await websocket.close(code=4401)  # custom code: unauthorized
            return

    await websocket.accept()
    print("[WS] Client connected")

    loop = asyncio.get_event_loop()

    try:
        while True:
            # 1. Pull latest annotated frame (non-blocking)
            b64_frame: Optional[str] = None
            try:
                b64_frame = await loop.run_in_executor(
                    None,
                    lambda: _frame_queue.get(timeout=0.12),
                )
            except queue.Empty:
                pass

            # 2. Snapshot of shared state
            with _state_lock:
                counts     = dict(_latest_counts)
                plates     = [
                    {"plate": p, "timestamp": int(time.time() * 1000)}
                    for p in _latest_plates.values()
                ]
                violations = list(_latest_violations)
                two_wheeler_statuses = list(_latest_two_wheeler_statuses)
                pedestrians = dict(_latest_pedestrians)
                fps        = round(_latest_fps, 1)
                src        = _source_type
                running    = _is_running
                video_done = _video_done
                video_sum  = _video_summary

            is_live_src = src in ("LIVE", "WEBCAM")

            # 3. Payload
            payload: Dict = {
                "fps":        fps,
                "counts":     counts,       # per-frame (live progress while processing)
                "plates":     plates,       # all plates seen so far
                "violations": violations,
                "two_wheeler_statuses": two_wheeler_statuses,
                "pedestrians": pedestrians,
                "source":     src,
                "running":    running,
                "mode":       "live" if is_live_src else "video",
                "video_done": video_done,   # True only when file finishes
            }
            if b64_frame:
                payload["frame"] = b64_frame

            # Send overall summary only when video is complete
            if video_done and video_sum:
                payload["video_summary"] = video_sum

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as exc:
        print(f"[WS] Error: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)