#!/usr/bin/env python3
"""
prepare_plate_data.py
Indian Road Intelligence System (IRIS) — Plate Dataset Preparation

Converts a raw license-plate dataset (Pascal VOC XML annotations, the
common format for Kaggle's "indian-license-plates-with-labels" set, or
pre-existing YOLO .txt annotations, the format used by most Roboflow
exports) into a single-class YOLOv8-ready train/val/test split with a
data.yaml, matching the layout expected by train_models.py.

Usage:
    # 1. Download & unzip your dataset first, e.g.:
    #    kaggle datasets download -d kedarsai/indian-license-plates-with-labels
    #    unzip indian-license-plates-with-labels.zip -d plate_dataset_raw

    # 2. Point this script at the raw folder. It auto-detects VOC XML vs
    #    YOLO .txt annotations sitting next to (or in an "annotations"
    #    subfolder of) the images.
    python prepare_plate_data.py --source plate_dataset_raw

    # 3. Optionally merge a second raw dataset in (e.g. a large generic
    #    Roboflow plate set) into the same split — run again with
    #    --source pointed at the second folder; images/labels accumulate.
    python prepare_plate_data.py --source roboflow_plate_dataset_raw

Then train with:
    python train_models.py --step train_plate
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is not installed. Run: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is not installed. Run: pip install tqdm")
    sys.exit(1)

# ==============================================================================
# CONFIG SECTION
# ==============================================================================
PLATE_TRAIN_DIR = "plate_dataset_train"
CLASS_NAME = "license_plate"
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# ==============================================================================


def find_images(source: Path) -> List[Path]:
    return sorted([p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS])


def find_label_for_image(img_path: Path, source: Path) -> Optional[Path]:
    """Look for a same-stem .xml (VOC) or .txt (YOLO) label file.
    Checks alongside the image, and in sibling 'annotations'/'labels' dirs."""
    candidates = [
        img_path.with_suffix(".xml"),
        img_path.with_suffix(".txt"),
        img_path.parent.parent / "annotations" / f"{img_path.stem}.xml",
        img_path.parent.parent / "labels" / f"{img_path.stem}.txt",
        source / "annotations" / f"{img_path.stem}.xml",
        source / "labels" / f"{img_path.stem}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def voc_xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> List[str]:
    """Parses a Pascal VOC XML file and returns YOLO-format label lines.
    All boxes are mapped to class 0 (license_plate) regardless of the
    original XML class name, since this is a single-class task."""
    lines = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return lines

    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
        except (AttributeError, ValueError, TypeError):
            continue

        xmin, xmax = sorted((max(0, xmin), min(img_w, xmax)))
        ymin, ymax = sorted((max(0, ymin), min(img_h, ymax)))
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def load_yolo_labels(txt_path: Path) -> List[str]:
    """Reads existing YOLO labels and forces every class index to 0,
    since multi-class source datasets (e.g. a generic Roboflow plate
    set with extra classes) still map to our single license_plate class."""
    lines = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = parts
            lines.append(f"0 {cx} {cy} {bw} {bh}")
    return lines


def convert_pair(img_path: Path, label_path: Path) -> Optional[List[str]]:
    if label_path.suffix.lower() == ".xml":
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            return None
        return voc_xml_to_yolo(label_path, w, h)
    elif label_path.suffix.lower() == ".txt":
        return load_yolo_labels(label_path)
    return None


def stage_pool(source: Path, pool_img_dir: Path, pool_lbl_dir: Path) -> Tuple[int, int]:
    pool_img_dir.mkdir(parents=True, exist_ok=True)
    pool_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(source)
    if not images:
        print(f"Warning: No images found under '{source}'.")
        return 0, 0

    converted, skipped = 0, 0
    print(f"\nScanning {len(images)} images under '{source}'...")
    for img_path in tqdm(images, desc="Converting annotations"):
        label_path = find_label_for_image(img_path, source)
        if label_path is None:
            skipped += 1
            continue

        yolo_lines = convert_pair(img_path, label_path)
        if not yolo_lines:
            skipped += 1
            continue

        # Use a source-prefixed filename to avoid collisions when merging
        # multiple raw datasets into the same pool.
        dest_stem = f"{source.name}_{img_path.stem}"
        dest_img = pool_img_dir / f"{dest_stem}{img_path.suffix.lower()}"
        dest_lbl = pool_lbl_dir / f"{dest_stem}.txt"

        shutil.copy2(img_path, dest_img)
        dest_lbl.write_text("\n".join(yolo_lines) + "\n")
        converted += 1

    return converted, skipped


def split_dataset(train_root: Path) -> None:
    pool_img_dir = train_root / "all_images"
    pool_lbl_dir = train_root / "all_labels"

    all_images = sorted(pool_img_dir.glob("*"))
    if not all_images:
        print("Error: No converted images in the pool. Nothing to split.")
        sys.exit(1)

    random.seed(42)
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    for split_name, split_imgs in splits.items():
        img_out = train_root / split_name / "images"
        lbl_out = train_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            lbl_path = pool_lbl_dir / f"{img_path.stem}.txt"
            shutil.copy2(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        print(f"  {split_name}: {len(split_imgs)} images")

    yaml_path = train_root / "data.yaml"
    yaml_path.write_text(
        f"path: {train_root.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    print(f"\nWrote dataset config to: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Indian license plate dataset for YOLOv8 training")
    parser.add_argument("--source", required=True, help="Path to a raw dataset folder (images + VOC XML or YOLO txt labels)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: source folder '{source}' does not exist.")
        sys.exit(1)

    train_root = Path(PLATE_TRAIN_DIR)
    pool_img_dir = train_root / "all_images"
    pool_lbl_dir = train_root / "all_labels"

    converted, skipped = stage_pool(source, pool_img_dir, pool_lbl_dir)
    print(f"\nConverted: {converted}   Skipped (no/invalid label): {skipped}")

    if converted == 0:
        print("\nError: No images were successfully converted. Check that your")
        print("dataset actually has VOC XML or YOLO txt annotations next to the images,")
        print("or in an 'annotations'/'labels' subfolder.")
        sys.exit(1)

    print("\nSplitting into train/val/test...")
    split_dataset(train_root)

    print("\n=======================================================")
    print("PLATE DATASET PREPARATION COMPLETE [OK]")
    print("=======================================================")
    print(f"Total usable images: {converted}")
    print("Next step, run:")
    print("  python train_models.py --step train_plate")
    print("=======================================================")


if __name__ == "__main__":
    main()
