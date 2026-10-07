#!/usr/bin/env python3
"""
Direct benchmark comparing Old (version 2.0) vs New (version 2.1) pedestrian pipeline
on identical video frames from real test clips.
Produces exact counts:
- Total unique pedestrians
- Males
- Females
- Children
- Unknown
- Rider false-positives
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Ensure backend in sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import config
from geometry_utils import calculate_iou, is_child_by_height, is_rider_of_vehicle
from pedestrian_detector import PedestrianDetector, DemographicClassifier, RawBox


def run_comparison(video_path: str, max_frames: int = 700):
    print(f"\n=======================================================")
    print(f"BENCHMARKING PIPELINE ON: {video_path}")
    print(f"=======================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    device = "cuda:0" if config.torch.cuda.is_available() else "cpu"

    # Vehicle tracker & model
    v_model = YOLO(config.VEHICLE_MODEL).to(device)
    
    # OLD PIPELINE SETUP (v2.0)
    old_gender_model = None
    old_model_path = Path("models/gender_detector.pt")
    if not old_model_path.exists():
        old_model_path = Path("backend/models/gender_detector.pt")
    if old_model_path.exists():
        old_gender_model = YOLO(str(old_model_path)).to(device)

    old_all_pedestrians_seen: Dict[int, str] = {}
    old_rider_fps = 0
    old_results: Dict[int, str] = {}

    # NEW PIPELINE SETUP (v2.1)
    new_detector = PedestrianDetector(device=device)
    new_rider_fps = 0

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        # Track vehicles
        v_results = v_model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=config.CONF_THRESHOLD,
            verbose=False,
        )
        vehicle_boxes = []
        vehicle_classes = []
        old_raw_results = []

        if v_results and v_results[0].boxes is not None:
            v_boxes = v_results[0].boxes
            if v_boxes.id is not None:
                v_xyxy = v_boxes.xyxy.cpu().numpy()
                v_cls_ids = v_boxes.cls.cpu().numpy().astype(int)
                v_track_ids = v_boxes.id.cpu().numpy().astype(int)
                for i in range(len(v_xyxy)):
                    cid = v_cls_ids[i]
                    cname = v_model.names.get(cid, "car")
                    box = (int(v_xyxy[i][0]), int(v_xyxy[i][1]), int(v_xyxy[i][2]), int(v_xyxy[i][3]))
                    vehicle_boxes.append(box)
                    vehicle_classes.append(cname)
                    old_raw_results.append(RawBox(cls=cid, track_id=v_track_ids[i], bbox=box))

        # ── 1. OLD PIPELINE PROCESSING ──────────────────────────
        # Old extracted persons via cls == 0 from vehicle model (which was Hatchback!)
        old_person_boxes = [b for b in old_raw_results if b.cls == 0]
        old_filtered = []
        for p in old_person_boxes:
            box_a = p.bbox
            is_ped = True
            for vb in vehicle_boxes:
                # Old IoU threshold 0.30
                if calculate_iou(box_a, vb) >= 0.30:
                    is_ped = False
                    break
            if is_ped:
                old_filtered.append(p)
            else:
                old_rider_fps += 1

        # Old classification & height heuristic
        all_heights = [p.bbox[3] - p.bbox[1] for p in old_filtered]
        avg_h = sum(all_heights) / len(all_heights) if all_heights else 0.0

        for p in old_filtered:
            tid = p.track_id
            crop = frame[max(0, p.bbox[1]):p.bbox[3], max(0, p.bbox[0]):p.bbox[2]]
            gender = "unknown"
            if old_gender_model and crop.size > 0:
                try:
                    res = old_gender_model.predict(crop, device=device, verbose=False)
                    if res and hasattr(res[0], 'probs') and res[0].probs is not None:
                        t1 = int(res[0].probs.top1)
                        cname = old_gender_model.names[t1].lower()
                        if "female" in cname:
                            gender = "female_adult"
                        elif "male" in cname:
                            gender = "male_adult"
                except Exception:
                    pass
            # Old naive height override
            ph = p.bbox[3] - p.bbox[1]
            if is_child_by_height(ph, avg_h, 0.60):
                gender = "child"

            old_results[tid] = gender
            if tid not in old_all_pedestrians_seen or gender != "unknown":
                old_all_pedestrians_seen[tid] = gender

        # ── 2. NEW PIPELINE PROCESSING ──────────────────────────
        new_pedestrians = new_detector.detect_and_track(frame, vehicle_boxes, vehicle_classes)
        for p in new_pedestrians:
            if new_detector.should_check(p.track_id, frame_idx):
                crop = frame[max(0, p.bbox[1]):p.bbox[3], max(0, p.bbox[0]):p.bbox[2]]
                new_detector.submit(
                    p.track_id,
                    crop,
                    "",
                    bbox=p.bbox,
                    frame_shape=frame.shape[:2],
                )
        new_detector.drain_completed()

    cap.release()

    # Tally Old Results
    old_totals = {"total": len(old_all_pedestrians_seen), "males": 0, "females": 0, "children": 0, "unknown": 0}
    for g in old_all_pedestrians_seen.values():
        if "male" in g:
            old_totals["males"] += 1
        elif "female" in g:
            old_totals["females"] += 1
        elif "child" in g:
            old_totals["children"] += 1
        else:
            old_totals["unknown"] += 1

    # Tally New Results
    new_totals = new_detector.get_session_summary()

    print(f"\n--- RESULTS FOR {Path(video_path).name} ({frame_idx} frames) ---")
    print(f"{'METRIC':<28} | {'OLD (v2.0)':<15} | {'NEW (v2.1)':<15}")
    print("-" * 64)
    print(f"{'Total Unique Pedestrians':<28} | {old_totals['total']:<15} | {new_totals['total']:<15}")
    print(f"{'Males':<28} | {old_totals['males']:<15} | {new_totals['males']:<15}")
    print(f"{'Females':<28} | {old_totals['females']:<15} | {new_totals['females']:<15}")
    print(f"{'Children':<28} | {old_totals['children']:<15} | {new_totals['children']:<15}")
    print(f"{'Unknown (Abstained)':<28} | {old_totals['unknown']:<15} | {new_totals['unknown']:<15}")
    print(f"{'Rider Exclusion Active':<28} | {'IoU >= 0.30':<15} | {'Area >= 0.50 + Up':<15}")
    print("=" * 64)

if __name__ == "__main__":
    import torch
    config.torch = torch
    target_video = sys.argv[1] if len(sys.argv) > 1 else "videos/L3.mp4"
    run_comparison(target_video)
