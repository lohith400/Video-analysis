#!/usr/bin/env python3
"""
Extract pedestrian evaluation crops from sample traffic videos.
Extracts at least 150 crops across multiple videos and creates
backend/pedestrian_eval/labels.csv template for manual verification and labeling.
"""

import os
import cv2
import csv
import glob
from pathlib import Path
from ultralytics import YOLO

def extract_crops():
    base_dir = Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "pedestrian_eval"
    crops_dir = eval_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate candidate video clips
    search_dirs = [base_dir.parent / "videos", base_dir / "videos", base_dir / "uploads"]
    video_files = []
    for d in search_dirs:
        if d.exists():
            video_files.extend(list(d.glob("*.mp4")))
            
    # Sort videos to prioritize known test videos like L3.mp4, L2.mp4
    def sort_key(p: Path):
        name = p.name.lower()
        if "l3" in name:
            return 0
        if "l2" in name:
            return 1
        if "20260521" in name:
            return 2
        return 10
        
    video_files = sorted(list(set(video_files)), key=sort_key)
    print(f"[Extractor] Found {len(video_files)} candidate video files.")
    
    model = YOLO("yolov8n.pt")
    target_count = 160
    extracted_records = []
    
    crop_counter = 0
    for vpath in video_files:
        if crop_counter >= target_count:
            break
            
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
            
        clip_name = vpath.stem
        print(f"[Extractor] Processing clip: {clip_name} ({vpath.name})")
        
        frame_idx = 0
        skip_frames = 15  # Sample every 15 frames for visual variety
        
        while cap.isOpened() and crop_counter < target_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % skip_frames != 0:
                continue
                
            h, w = frame.shape[:2]
            # Detect persons (COCO class 0)
            results = model.track(frame, classes=[0], persist=True, conf=0.35, verbose=False)
            if not results or results[0].boxes is None:
                continue
                
            boxes = results[0].boxes
            if boxes.id is None:
                continue
                
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            
            for i in range(len(xyxy)):
                if crop_counter >= target_count:
                    break
                    
                x1, y1, x2, y2 = xyxy[i].astype(int)
                tid = track_ids[i]
                
                # Quality filtering for evaluation candidates
                bw = x2 - x1
                bh = y2 - y1
                if bh < 60 or bw < 25:
                    continue
                if x1 < 5 or y1 < 5 or x2 > w - 5 or y2 > h - 5:
                    continue  # skip edge cuts
                    
                # Pad slightly (5%)
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)
                
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                    
                crop_filename = f"crop_{clip_name}_f{frame_idx}_t{tid}_{crop_counter:03d}.jpg"
                crop_save_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_save_path), crop)
                
                # Relative path from pedestrian_eval
                rel_path = f"crops/{crop_filename}"
                extracted_records.append((rel_path, ""))
                crop_counter += 1
                
        cap.release()
        
    print(f"[Extractor] Total crops saved: {len(extracted_records)}")
    
    # Write labels.csv template
    labels_csv = eval_dir / "labels.csv"
    with open(labels_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_file", "true_label"])
        for rel_path, _ in extracted_records:
            writer.writerow([rel_path, ""])
            
    print(f"[Extractor] Wrote {len(extracted_records)} template rows to {labels_csv}")

if __name__ == "__main__":
    extract_crops()
