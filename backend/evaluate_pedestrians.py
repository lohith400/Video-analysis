#!/usr/bin/env python3
"""
Pedestrian Demographic Evaluation Suite.
Evaluates accuracy, precision, recall, confusion matrix, abstain rate,
and track-level voting against manual annotations in backend/pedestrian_eval/labels.csv.
Saves detailed evaluation report to Notes/pedestrian_eval_results.md.
"""

import os
import sys
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np

# Ensure backend directory is in path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import config
from pedestrian_detector import PedestrianDetector, DemographicClassifier


def run_evaluation():
    eval_dir = backend_dir / "pedestrian_eval"
    labels_file = eval_dir / "labels.csv"
    results_md = backend_dir.parent / "Notes" / "pedestrian_eval_results.md"
    results_md.parent.mkdir(parents=True, exist_ok=True)

    if not labels_file.exists():
        print(f"[Evaluate] ERROR: Labels file not found at {labels_file}")
        return

    # Read CSV
    labeled_data = []
    unlabeled_count = 0
    with open(labels_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_rel = row.get("crop_file", "").strip()
            true_lbl = row.get("true_label", "").strip().lower()
            if not true_lbl:
                unlabeled_count += 1
            else:
                labeled_data.append((crop_rel, true_lbl))

    print(f"[Evaluate] Loaded {len(labeled_data)} labeled crops ({unlabeled_count} pending labels).")

    if len(labeled_data) == 0:
        msg = f"## Pedestrian Evaluation Status\n\n- Total evaluation candidates: {unlabeled_count}\n- Labeled crops: 0\n- Status: **Labelling needed**: open `backend/pedestrian_eval/labels.csv` and fill `true_label` with `male`, `female`, `child`, or `unsure`.\n"
        print(msg)
        with open(results_md, mode="w", encoding="utf-8") as f:
            f.write(msg)
        return

    # Initialize demographic classifier
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
    except Exception:
        pass

    classifier = DemographicClassifier(device=device)

    valid_classes = ["male", "female", "child"]
    confusion = defaultdict(lambda: defaultdict(int))
    track_predictions = defaultdict(list)
    track_ground_truth = {}

    total_valid = 0
    correct_valid = 0
    abstain_count = 0

    for crop_rel, true_label in labeled_data:
        crop_path = eval_dir / crop_rel
        if not crop_path.exists():
            continue

        img = cv2.imread(str(crop_path))
        if img is None:
            continue

        # Extract track id if present in filename: e.g., t12_
        m = re.search(r"_t(\d+)_", crop_path.name)
        tid = int(m.group(1)) if m else -1

        pred_label, conf = classifier.classify(img)
        # Normalize labels
        if pred_label in ["male_adult", "man", "boy"]:
            pred_label = "male"
        elif pred_label in ["female_adult", "woman", "girl"]:
            pred_label = "female"
        elif pred_label in ["child", "kid"]:
            pred_label = "child"
        elif pred_label not in ["male", "female", "child"]:
            pred_label = "unknown"

        if pred_label == "unknown":
            abstain_count += 1

        if true_label in valid_classes:
            total_valid += 1
            confusion[true_label][pred_label] += 1
            if pred_label == true_label:
                correct_valid += 1

        if tid >= 0:
            track_predictions[tid].append((pred_label, conf))
            if true_label in valid_classes:
                track_ground_truth[tid] = true_label

    acc = (correct_valid / total_valid * 100.0) if total_valid > 0 else 0.0
    abstain_rate = (abstain_count / len(labeled_data) * 100.0) if labeled_data else 0.0

    # Per-class metrics
    per_class = {}
    for cls in valid_classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in valid_classes if other != cls)
        fn = sum(confusion[cls][other] for other in (valid_classes + ["unknown"]) if other != cls)
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fn": fn, "fp": fp}

    # Track-level evaluation after temporal voting
    track_correct = 0
    track_total = 0
    for tid, true_lbl in track_ground_truth.items():
        votes = track_predictions[tid]
        # Weighted voting
        scores = defaultdict(float)
        for p_lbl, c in votes:
            if p_lbl in valid_classes:
                scores[p_lbl] += c
        if scores:
            voted_lbl = max(scores.items(), key=lambda x: x[1])[0]
        else:
            voted_lbl = "unknown"

        track_total += 1
        if voted_lbl == true_lbl:
            track_correct += 1

    track_acc = (track_correct / track_total * 100.0) if track_total > 0 else 0.0

    # Build Markdown Output
    report = []
    report.append("# Pedestrian Demographic Evaluation Results\n")
    report.append(f"- **Evaluated Samples**: {len(labeled_data)}")
    report.append(f"- **Valid Evaluated Ground Truths**: {total_valid}")
    report.append(f"- **Overall Crop Accuracy**: {acc:.2f}%")
    report.append(f"- **Abstain Rate (Unknown / Gated)**: {abstain_rate:.2f}%")
    report.append(f"- **Track-Level Accuracy After Voting**: {track_acc:.2f}%\n")

    report.append("### Per-Class Precision & Recall\n")
    report.append("| Class | Precision | Recall | F1-Score | Support |")
    report.append("|---|---|---|---|---|")
    for cls in valid_classes:
        m = per_class[cls]
        supp = m["tp"] + m["fn"]
        report.append(f"| **{cls.capitalize()}** | {m['precision']:.1f}% | {m['recall']:.1f}% | {m['f1']:.1f}% | {supp} |")
    report.append("\n")

    report.append("### Confusion Matrix\n")
    report.append("| True \\ Pred | Male | Female | Child | Unknown |")
    report.append("|---|---|---|---|---|")
    for cls in valid_classes:
        report.append(f"| **{cls.capitalize()}** | {confusion[cls]['male']} | {confusion[cls]['female']} | {confusion[cls]['child']} | {confusion[cls]['unknown']} |")
    report.append("\n")

    report_str = "\n".join(report)
    print("\n" + report_str)

    with open(results_md, mode="w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[Evaluate] Results saved to {results_md}")

if __name__ == "__main__":
    run_evaluation()
