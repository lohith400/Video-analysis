#!/usr/bin/env python3
"""
prepare_data.py
Indian Road Intelligence System (IRIS) — Data Engineering Script

Orchestrates frame extraction from videos and train/val/test dataset split configurations.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is not installed. Run: pip install tqdm")
    sys.exit(1)

# ==============================================================================
# CONFIG SECTION
# ==============================================================================
VIDEO_DIR = "videos"
HELMET_IMAGE_DIR = "helmet_dataset/images/all"
HELMET_LABEL_DIR = "helmet_dataset/labels/all"
GENDER_IMAGE_DIR = "gender_dataset/images/all"
GENDER_LABEL_DIR = "gender_dataset/labels/all"
HELMET_TRAIN_DIR = "helmet_dataset_train"
GENDER_TRAIN_DIR = "gender_dataset_train"

HELMET_FRAME_SKIP = 6
GENDER_FRAME_SKIP = 15
MAX_HELMET_FRAMES_PER_VIDEO = 400
MAX_GENDER_FRAMES_PER_VIDEO = 300

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
# ==============================================================================


def get_videos() -> List[Path]:
    video_path = Path(VIDEO_DIR)
    if not video_path.exists():
        print(f"Warning: Video directory '{VIDEO_DIR}' does not exist. Creating it.")
        video_path.mkdir(parents=True, exist_ok=True)
        return []
    
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [p for p in video_path.iterdir() if p.suffix.lower() in valid_exts]
    return sorted(videos)


def extract_frames(mode: str) -> None:
    videos = get_videos()
    if not videos:
        print(f"\nNo video files found in '{VIDEO_DIR}/'. Place your Indian traffic videos there first.")
        return

    if mode == "helmet":
        dest_dir = Path(HELMET_IMAGE_DIR)
        skip = HELMET_FRAME_SKIP
        max_frames = MAX_HELMET_FRAMES_PER_VIDEO
    else:
        dest_dir = Path(GENDER_IMAGE_DIR)
        skip = GENDER_FRAME_SKIP
        max_frames = MAX_GENDER_FRAMES_PER_VIDEO

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nScanning '{VIDEO_DIR}/' for video files...")
    print(f"Found {len(videos)} videos: {[v.name for v in videos]}")
    print(f"Extracting frames to: {dest_dir} (skip={skip}, max_limit={max_frames} per video)")

    total_extracted = 0

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video.name}. Skipping.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        base_name = video.stem
        
        extracted_this_video = 0
        frame_idx = 0
        saved_count = 0

        # tqdm progress bar
        with tqdm(total=min(total_frames, max_frames * skip), desc=f"Video: {video.name}", unit="fr") as pbar:
            while cap.isOpened() and saved_count < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % skip == 0:
                    saved_count += 1
                    out_filename = dest_dir / f"{base_name}_{saved_count:04d}.jpg"
                    cv2.imwrite(str(out_filename), frame)
                    extracted_this_video += 1
                    total_extracted += 1

                    if extracted_this_video % 50 == 0:
                        tqdm.write(f"[{video.name}] Extracted {extracted_this_video} frames...")

                frame_idx += 1
                pbar.update(1)

        cap.release()
        print(f"Finished video {video.name}: Saved {extracted_this_video} images.")

    print(f"\nDone! Total images extracted across all videos: {total_extracted}")
    
    if mode == "helmet":
        print_helmet_instructions()
    else:
        print_gender_instructions()


def print_helmet_instructions() -> None:
    print("""
  +----------------------------------------------------------+
  |           ANNOTATION INSTRUCTIONS                        |
  +----------------------------------------------------------+
  |  1. Open terminal and run: anylabeling                   |
  |  2. File -> Open Dir -> select helmet_dataset/images/all/|
  |  3. Click AI button -> select YOLO-World model            |
  |  4. In class box type your 4 classes one by one:         |
  |       rider_helmet                                       |
  |       rider_no_helmet                                    |
  |       pillion_helmet                                     |
  |       pillion_no_helmet                                  |
  |  5. Click Auto Annotate on each image                    |
  |  6. Verify boxes - fix any wrong ones manually           |
  |  7. File -> Export -> YOLO format                         |
  |  8. Save labels to: helmet_dataset/labels/all/           |
  |  9. When done run:                                       |
  |       python prepare_data.py --step setup_helmet         |
  +----------------------------------------------------------+
    """)


def print_gender_instructions() -> None:
    print("""
  +----------------------------------------------------------+
  |           ANNOTATION INSTRUCTIONS                        |
  +----------------------------------------------------------+
  |  1. Open terminal and run: anylabeling                   |
  |  2. File -> Open Dir -> select gender_dataset/images/all/|
  |  3. Click AI button -> select YOLO-World model            |
  |  4. In class box type your 3 classes one by one:         |
  |       male_adult                                         |
  |       female_adult                                       |
  |       child                                              |
  |  5. Click Auto Annotate on each image                    |
  |  6. Verify boxes - fix any wrong ones manually           |
  |  7. File -> Export -> YOLO format                         |
  |  8. Save labels to: gender_dataset/labels/all/           |
  |  9. When done run:                                       |
  |       python prepare_data.py --step setup_gender         |
  +----------------------------------------------------------+
    """)


def setup_dataset(mode: str) -> None:
    if mode == "helmet":
        img_src = Path(HELMET_IMAGE_DIR)
        lbl_src = Path(HELMET_LABEL_DIR)
        train_dir = Path(HELMET_TRAIN_DIR)
        classes = ["rider_helmet", "rider_no_helmet", "pillion_helmet", "pillion_no_helmet"]
    else:
        img_src = Path(GENDER_IMAGE_DIR)
        lbl_src = Path(GENDER_LABEL_DIR)
        train_dir = Path(GENDER_TRAIN_DIR)
        classes = ["male_adult", "female_adult", "child"]

    if not img_src.exists():
        print(f"Error: Source images directory '{img_src}' does not exist. Run frame extraction first.")
        return

    if not lbl_src.exists():
        print(f"Error: Source labels directory '{lbl_src}' does not exist.")
        print(f"Please annotate your images inside X-AnyLabeling and export them to '{lbl_src}'.")
        return

    # Gather matching images and labels
    all_images = sorted([p for p in img_src.iterdir() if p.suffix.lower() == ".jpg"])
    all_labels = sorted([p for p in lbl_src.iterdir() if p.suffix.lower() == ".txt"])

    print(f"\nCounting files in source directories:")
    print(f"Total Images: {len(all_images)}")
    print(f"Total Labels: {len(all_labels)}")

    if len(all_images) == 0:
        print("Error: No images found to split.")
        return

    # Check for missing label files
    image_names = {img.stem for img in all_images}
    label_names = {lbl.stem for lbl in all_labels}
    
    missing_labels = image_names - label_names
    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} images are missing matching annotation files.")
        print("Missing label files for these images:")
        for name in list(missing_labels)[:10]:
            print(f"  - {name}.jpg")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more.")

    # We split based on valid matched image-label pairs to prevent YOLO training crashes
    matched_stems = sorted(list(image_names & label_names))
    if not matched_stems:
        print("\nError: No matching image-label pairs found!")
        print(f"Make sure you saved YOLO label text files inside: {lbl_src}")
        return

    print(f"\nMatched image-label pairs ready for split: {len(matched_stems)}")

    # Split directories creation
    splits = ["train", "valid", "test"]
    for split in splits:
        (train_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (train_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Perform split calculation
    import random
    random.seed(42)
    random.shuffle(matched_stems)

    total = len(matched_stems)
    n_train = int(total * TRAIN_SPLIT)
    n_val = int(total * VAL_SPLIT)
    
    train_stems = matched_stems[:n_train]
    val_stems = matched_stems[n_train:n_train + n_val]
    test_stems = matched_stems[n_train + n_val:]

    def copy_split_files(stems: List[str], split_name: str) -> int:
        copied = 0
        for stem in stems:
            img_file = img_src / f"{stem}.jpg"
            lbl_file = lbl_src / f"{stem}.txt"
            
            if img_file.exists() and lbl_file.exists():
                shutil.copy2(img_file, train_dir / split_name / "images" / f"{stem}.jpg")
                shutil.copy2(lbl_file, train_dir / split_name / "labels" / f"{stem}.txt")
                copied += 1
        return copied

    print("\nCopying split files into train/val/test splits...")
    train_copied = copy_split_files(train_stems, "train")
    val_copied = copy_split_files(val_stems, "valid")
    test_copied = copy_split_files(test_stems, "test")

    # Generate data.yaml with absolute path
    abs_train_path = os.path.abspath(train_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_train_path}
train: train/images
val: valid/images
test: test/images

nc: {len(classes)}
names:
"""
    for name in classes:
        yaml_content += f"  - {name}\n"

    yaml_file = train_dir / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n=======================================================")
    print(f"DATASET READY: {train_dir.name}")
    print(f"=======================================================")
    print(f"Train images: {train_copied}")
    print(f"Val images:   {val_copied}")
    print(f"Test images:  {test_copied}")
    print(f"Generated YAML at: {yaml_file}")
    print(f"Absolute Path Configuration: {abs_train_path}")
    print(f"=======================================================")
    print(f"Next step, run:")
    print(f"  python train_models.py --step train_{mode}")
    print(f"=======================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IRIS Road Intelligence System — Data Engineering Scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Steps:
  extract_helmet   Extracts every 6th frame from videos/ for helmet dataset.
  extract_gender   Extracts every 15th frame from videos/ for gender dataset.
  setup_helmet     Combines images and labels, splits 80/10/10, and generates data.yaml.
  setup_gender     Combines images and labels, splits 80/10/10, and generates data.yaml.
        """
    )
    parser.add_argument(
        "--step",
        choices=["extract_helmet", "extract_gender", "setup_helmet", "setup_gender"],
        help="Specify the pipeline step to execute."
    )

    args = parser.parse_args()

    if not args.step:
        parser.print_help()
        sys.exit(0)

    if args.step == "extract_helmet":
        extract_frames("helmet")
    elif args.step == "extract_gender":
        extract_frames("gender")
    elif args.step == "setup_helmet":
        setup_dataset("helmet")
    elif args.step == "setup_gender":
        setup_dataset("gender")


if __name__ == "__main__":
    main()
