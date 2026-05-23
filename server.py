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

try:
    from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Run: pip install fastapi uvicorn python-multipart")

import config
from annotator import count_by_category, draw_annotations
from detector import VehicleModelLoader, crop_vehicle, get_device
from tracker import VehicleTracker

# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="Indian Road Intelligence System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
_pipeline_lock = threading.Lock()

_analysis_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_pipeline() -> None:
    global _loader, _tracker, _ocr
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
    global _video_done, _video_summary
    with _state_lock:
        _video_done    = False
        _video_summary = None


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
                _latest_fps    = fps

            # ── Annotate + push frame ───────────────────────────────────────
            try:
                total_plates = _ocr.total_plates_detected if _ocr else 0
                annotated    = draw_annotations(
                    frame, vehicles, plates_map, counts, fps, total_plates, counter=counter, plate_boxes=plate_boxes_map
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

            with _state_lock:
                _video_done    = True
                _video_summary = {"counts": final_counts, "plates": final_plates}

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


@app.post("/upload")
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


@app.post("/connect")
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


@app.post("/stop")
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