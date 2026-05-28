"""
generate_training_report.py
----------------------------
Run this AFTER training completes to generate a full training report README
inside the plate_dataset/indian-plate-detector.yolov8/ folder.

Usage (from backend/ directory):
    ../.venv/Scripts/python generate_training_report.py
"""

import csv
import os
import glob
from datetime import datetime
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
BACKEND_DIR     = Path(__file__).parent
PROJECT_ROOT    = BACKEND_DIR.parent
RUNS_DIR        = PROJECT_ROOT / "runs" / "detect"   # YOLO saves here (project root)
TRAINING_NAME   = "indian_plate_v2"
DATASET_DIR     = BACKEND_DIR / "plate_dataset" / "indian-plate-detector.yolov8"
REPORT_OUTPUT   = DATASET_DIR / "TRAINING_REPORT.md"
# ────────────────────────────────────────────────────────────────────────────


def count_files(folder: Path, extensions=(".jpg", ".jpeg", ".png")):
    """Count image files in a folder recursively."""
    if not folder.exists():
        return 0
    return sum(1 for ext in extensions for _ in folder.rglob(f"*{ext}"))


def load_results_csv(run_dir: Path):
    """Parse YOLO results.csv into a list of dicts."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def find_best_epoch(rows):
    """Find the epoch with the best mAP50."""
    best = None
    best_map = -1.0
    for row in rows:
        try:
            val = float(row.get("metrics/mAP50(B)", 0))
            if val > best_map:
                best_map = val
                best = row
        except ValueError:
            continue
    return best


def rating(map50):
    """Human-readable quality rating based on mAP50."""
    if map50 >= 0.90:
        return "[EXCELLENT]  Production ready"
    elif map50 >= 0.80:
        return "[VERY GOOD]  Ready for deployment"
    elif map50 >= 0.70:
        return "[GOOD]       Acceptable, could improve with more data"
    elif map50 >= 0.55:
        return "[FAIR]       Needs more data or longer training"
    else:
        return "[POOR]       Needs significant improvement"


def main():
    run_dir = RUNS_DIR / TRAINING_NAME

    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        print("   Make sure training completed and TRAINING_NAME is correct.")
        return

    rows = load_results_csv(run_dir)
    if not rows:
        print("[ERROR] results.csv not found or empty.")
        return

    last   = rows[-1]
    best   = find_best_epoch(rows)
    total_epochs = len(rows)

    # ── Dataset stats ────────────────────────────────────────────────────
    train_images = count_files(DATASET_DIR / "train" / "images")
    valid_images = count_files(DATASET_DIR / "valid" / "images")
    test_images  = count_files(DATASET_DIR / "test"  / "images")
    total_images = train_images + valid_images + test_images

    # ── Best weights path ─────────────────────────────────────────────────
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    # ── Metrics ───────────────────────────────────────────────────────────
    def fmt(row, key, pct=True):
        try:
            val = float(row[key])
            return f"{val*100:.2f}%" if pct else f"{val:.4f}"
        except (KeyError, ValueError):
            return "N/A"

    best_map50    = float(best.get("metrics/mAP50(B)", 0))    if best else 0
    best_map5095  = float(best.get("metrics/mAP50-95(B)", 0)) if best else 0
    best_precision= float(best.get("metrics/precision(B)", 0))if best else 0
    best_recall   = float(best.get("metrics/recall(B)", 0))   if best else 0
    best_epoch_no = best.get("                  epoch", best.get("epoch", "?")) if best else "?"

    # ── Report ────────────────────────────────────────────────────────────
    report = f"""# 🚗 Indian License Plate Detector — Training Report

> **Generated:** {datetime.now().strftime("%d %B %Y, %I:%M %p")}
> **Model:** YOLOv8n (fine-tuned via transfer learning from COCO weights)
> **Project:** Indian Road Intelligence System — ANPR Module

---

## 📊 Dataset Summary

| Property         | Value                          |
|------------------|-------------------------------|
| Dataset Name     | indian-plate-detector          |
| Source           | Roboflow (lohiths-workspace)   |
| Classes          | 1 — `license_plate`            |
| Total Images     | {total_images if total_images > 0 else train_images} |
| Train Split      | {train_images} images          |
| Valid Split      | {valid_images if valid_images > 0 else "Same as Train"} images |
| Test Split       | {test_images if test_images > 0 else "Same as Train"} images  |
| Image Size       | 640 × 640 px                   |

---

## ⚙️ Training Configuration

| Parameter        | Value                          |
|------------------|-------------------------------|
| Base Model       | yolov8n.pt (COCO pretrained)   |
| Total Epochs     | {total_epochs}                 |
| Batch Size       | 8                              |
| Precision        | FP16 Mixed (AMP enabled)       |
| Device           | NVIDIA RTX 3050 Laptop GPU     |
| VRAM             | ~6 GB                          |
| Workers          | 0 (Windows compatible)         |
| Optimizer        | AdamW (YOLO default)           |

---

## 🏅 Best Epoch Results (Epoch {best_epoch_no})

| Metric                  | Value                        |
|-------------------------|------------------------------|
| **mAP@50**              | **{best_map50*100:.2f}%**    |
| **mAP@50-95**           | **{best_map5095*100:.2f}%**  |
| **Precision**           | **{best_precision*100:.2f}%**|
| **Recall**              | **{best_recall*100:.2f}%**   |

### 🔍 What These Mean:
- **Precision** — Of all detections made, how many were actually license plates?
- **Recall** — Of all real license plates in the image, how many did the model find?
- **mAP@50** — Mean Average Precision at 50% IoU overlap threshold (primary metric)
- **mAP@50-95** — Stricter average across 50–95% IoU thresholds (tighter bounding boxes)

---

## 📈 Final Epoch Results (Epoch {last.get('                  epoch', last.get('epoch', total_epochs))})

| Metric                  | Value                        |
|-------------------------|------------------------------|
| mAP@50                  | {fmt(last, 'metrics/mAP50(B)')}    |
| mAP@50-95               | {fmt(last, 'metrics/mAP50-95(B)')} |
| Precision               | {fmt(last, 'metrics/precision(B)')}|
| Recall                  | {fmt(last, 'metrics/recall(B)')}   |
| Box Loss                | {fmt(last, 'train/box_loss', pct=False)} |
| Class Loss              | {fmt(last, 'train/cls_loss', pct=False)} |

---

## ⭐ Overall Model Quality

> **{rating(best_map50)}**

---

## 📂 Output Files

| File                                              | Description                              |
|---------------------------------------------------|------------------------------------------|
| `{best_weights}`  | ✅ Best weights (use this in production) |
| `{last_weights}`  | Last checkpoint weights                  |
| `{run_dir}/results.csv`             | Full per-epoch metrics log               |
| `{run_dir}/confusion_matrix.png`    | Confusion matrix plot                    |
| `{run_dir}/PR_curve.png`            | Precision-Recall curve                   |
| `{run_dir}/results.png`             | Loss & metric curves chart               |

---

## 🚀 How to Deploy

After training, copy the best weights to the production models folder:

```powershell
copy runs\\detect\\indian_plate_v2\\weights\\best.pt models\\license_plate_detector.pt
```

The `license_plate_detector.pt` file is loaded by `server.py` and `ocr_engine.py`
to detect license plates in real-time video streams.

---

## 📉 Full Epoch-by-Epoch Log

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
"""

    for row in rows:
        ep  = row.get("                  epoch", row.get("epoch", "?")).strip()
        p   = fmt(row, "metrics/precision(B)")
        r   = fmt(row, "metrics/recall(B)")
        m50 = fmt(row, "metrics/mAP50(B)")
        m95 = fmt(row, "metrics/mAP50-95(B)")
        report += f"| {ep:>5} | {p:>9} | {r:>6} | {m50:>6} | {m95:>9} |\n"

    report += f"""
---

*Report auto-generated by `generate_training_report.py` · Indian Road Intelligence System*
"""

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(report, encoding="utf-8")
    print(f"\n[DONE] Report saved to: {REPORT_OUTPUT.resolve()}")
    print(f"   Best mAP@50 : {best_map50*100:.2f}%")
    print(f"   Best Epoch  : {best_epoch_no}")
    print(f"   Rating      : {rating(best_map50)}")


if __name__ == "__main__":
    main()
