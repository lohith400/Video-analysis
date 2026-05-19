#!/usr/bin/env python3
"""
FastAPI backend for Indian Road Intelligence System.
Streams annotated frames over WebSocket as base64-encoded JPEG.
OCR / plate detection DISABLED — vehicle detection only.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
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

try:
    from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Run: pip install fastapi uvicorn python-multipart")

import config
from annotator import count_by_category, draw_annotations
from csv_logger import CSVLogger
from detector import VehicleModelLoader, crop_vehicle, get_device
from ocr_engine import OCREngine   # stub — returns empty results
from tracker import VehicleTracker

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Indian Road Intelligence System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ───────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_latest_counts: Dict = {c: 0 for c in config.ALL_VEHICLE_CLASSES}
_latest_counts["total"] = 0
_latest_plates: Dict[int, str] = {}
_latest_violations: List[Dict] = []
_latest_fps: float = 0.0
_source_type: str = "VIDEO"

# Frame queue — maxsize=2 so we always serve freshest frame
_frame_queue: queue.Queue = queue.Queue(maxsize=2)

# Pipeline singletons (lazy-loaded once on first upload/connect)
_device = get_device()
_loader: Optional[VehicleModelLoader] = None
_tracker: Optional[VehicleTracker] = None
_ocr: Optional[OCREngine] = None
_pipeline_lock = threading.Lock()

# Thread control
_analysis_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# ── Pipeline loader ────────────────────────────────────────────────────────
def _load_pipeline() -> None:
    global _loader, _tracker, _ocr
    with _pipeline_lock:
        if _loader is not None:
            return
        print(f"[server] Loading vehicle model on {_device} …")
        _loader = VehicleModelLoader(_device)
        _tracker = VehicleTracker(_loader.yolo, _device, _loader.vehicle_class_ids)
        _ocr = OCREngine(_device)   # stub — no plate model needed
        print("[server] Models ready. Plate detection: DISABLED")


def _frame_to_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _flush_frame_queue() -> None:
    while not _frame_queue.empty():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            break


# ── Analysis thread ────────────────────────────────────────────────────────
def _run_analysis(source, source_type: str) -> None:
    global _latest_counts, _latest_fps, _source_type

    _load_pipeline()
    _source_type = source_type
    _stop_event.clear()
    _flush_frame_queue()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[server] Cannot open source: {source}")
        return

    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    csv_logger = CSVLogger(
        counts_getter=lambda: dict(_latest_counts),
        plates_getter=lambda: {},   # always empty — OCR disabled
    )
    csv_logger.start()

    fps = 0.0
    t_prev = time.perf_counter()
    frame_idx = 0
    print(f"[server] Analysis started: source={source!r} type={source_type}")

    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and source.lower().startswith("rtsp://"):
                    print("[server] RTSP lost, reconnecting…")
                    cap.release()
                    time.sleep(config.RTSP_RECONNECT_WAIT_SEC)
                    cap = cv2.VideoCapture(source)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                break  # EOF for video file

            # ── Track vehicles ───────────────────────────────────────────
            vehicles = _tracker.track(frame)
            counts = count_by_category(vehicles)

            # ── OCR DISABLED — skip entirely ────────────────────────────
            # No plate submission, no drain needed

            # ── FPS ──────────────────────────────────────────────────────
            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt
            t_prev = t_now

            # ── Update shared state ──────────────────────────────────────
            with _state_lock:
                _latest_counts = counts
                _latest_fps = fps

            # ── Annotate frame ───────────────────────────────────────────
            annotated = draw_annotations(
                frame,
                vehicles,
                {},       # empty plate map
                counts,
                fps,
                0,        # total plates = 0
            )

            # ── Encode and push to WebSocket queue ───────────────────────
            b64 = _frame_to_b64(annotated)
            if b64:
                if _frame_queue.full():
                    try:
                        _frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    _frame_queue.put_nowait(b64)
                except queue.Full:
                    pass

            frame_idx += 1

    except Exception as exc:
        print(f"[server] Analysis error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        csv_logger.stop()
        print("[server] Analysis stopped.")


def _stop_analysis(timeout: float = 3.0) -> None:
    global _analysis_thread
    _stop_event.set()
    if _analysis_thread and _analysis_thread.is_alive():
        _analysis_thread.join(timeout=timeout)
    _analysis_thread = None


# ── REST endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": _device,
        "models_loaded": _loader is not None,
        "ocr_enabled": False,
        "analysis_running": bool(_analysis_thread and _analysis_thread.is_alive()),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global _analysis_thread

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / file.filename
    dest.write_bytes(await file.read())

    mime = file.content_type or ""
    source_type = "VIDEO" if mime.startswith("video") else "IMAGE"

    _stop_analysis()
    _analysis_thread = threading.Thread(
        target=_run_analysis,
        args=(str(dest), source_type),
        daemon=True,
    )
    _analysis_thread.start()
    return {"status": "started", "file": file.filename, "source_type": source_type}


class ConnectRequest(BaseModel):
    source: str


@app.post("/connect")
async def connect(req: ConnectRequest):
    global _analysis_thread
    raw = req.source.strip()
    source = 0 if raw.lower() == "webcam" else raw
    source_type = "WEBCAM" if raw.lower() == "webcam" else "LIVE"

    _stop_analysis()
    _analysis_thread = threading.Thread(
        target=_run_analysis,
        args=(source, source_type),
        daemon=True,
    )
    _analysis_thread.start()
    return {"status": "connecting", "source": req.source}


@app.post("/stop")
async def stop():
    _stop_analysis()
    return {"status": "stopped"}


@app.get("/analytics")
async def analytics():
    csv_path = Path(config.CSV_PATH)
    if not csv_path.exists():
        return {"rows": []}
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return {"rows": rows}


# ── WebSocket: stream frames + metadata ───────────────────────────────────
@app.websocket("/video-feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    loop = asyncio.get_event_loop()

    try:
        while True:
            b64_frame: Optional[str] = None
            try:
                b64_frame = await loop.run_in_executor(
                    None,
                    lambda: _frame_queue.get(timeout=0.15),
                )
            except queue.Empty:
                pass

            with _state_lock:
                counts = dict(_latest_counts)
                fps    = _latest_fps
                src    = _source_type

            payload: Dict = {
                "fps":        round(fps, 1),
                "counts":     counts,
                "plates":     [],       # always empty — OCR disabled
                "violations": [],       # always empty — OCR disabled
                "source":     src,
            }
            if b64_frame:
                payload["frame"] = b64_frame

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as exc:
        print(f"[WS] Error: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)