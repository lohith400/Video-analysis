# Model Card — IRIS Custom-Trained Models

Three custom YOLOv8-Nano models were trained for this project. All
figures below are taken from actual training runs, not estimated.
**Every dataset here is small and single-source** — see
[LIMITATIONS.md](LIMITATIONS.md) for what that means for how much to
trust these numbers.

## 1. License Plate Detector

| Property | Value |
|---|---|
| Base architecture | YOLOv8-Nano (COCO-pretrained) |
| Parameters | 3.01 M — 5.94 MB weights |
| Training run | `indian_plate_v2`, 50 epochs, batch 8, AdamW, 640×640 input |
| Training time | 1,724.6 s (≈28.7 min) on RTX 3050 |
| mAP@50 | 95.35% |
| mAP@50–95 | 66.97% |
| Precision / Recall | 84.51% / 93.32% |
| Final validation losses | Box 0.985, Cls 0.567, DFL 0.801 |
| Dataset | 320 images, Roboflow-hosted annotation project + a legacy supplemental set |
| Held-out test split | **None reported** for the active 320-image set |

The gap between mAP@50 (95.4%) and mAP@50-95 (67.0%) means the model
finds plates reliably but bounding boxes aren't always tightly
localised — typical for a small nano model, and acceptable here because
the downstream OCR crop is padded before recognition.

## 2. Helmet Compliance Detector

| Property | Value |
|---|---|
| Base architecture | YOLOv8-Nano |
| Parameters | 3.16 M — 6.25 MB weights |
| Classes | `with_helmet`, `without_helmet`, `head` (bare) |
| Runtime | Async, confidence 0.45, re-checked every 5 frames, capped at 15 attempts/track |
| Dataset | 210 hand-collected, self-annotated images (test split: 21 images) |
| Precision / Recall | **Not recorded** — flagged as an open item |

Trained on a very small dataset. Expected to generalise poorly to
headgear, clothing, and lighting conditions outside the collected
footage. Functional prototype, not a production classifier.

## 3. Pedestrian Demographics Classifier

| Property | Value |
|---|---|
| Base architecture | YOLOv8-Nano |
| Parameters | 3.16 M — 6.25 MB weights |
| Classes | `male_adult`, `female_adult`, `child` |
| Fallback | DeepFace-based estimator, used only if custom weights are unavailable at runtime |
| Dataset | 84 hand-collected, self-annotated images (test split: 9 images) |

Child classification is reinforced by a rule-based, scene-relative
height heuristic (`geometry_utils.is_child_by_height`) rather than
relying on the classifier alone — a deliberate compensation for a
training set too small to trust for a subtle category like child vs.
short adult.

## Baseline / supporting weights

| Weight file | Params | Size | Role |
|---|---|---|---|
| `yolov8n.pt` | 3.16 M | 6.25 MB | Base vehicle/person detector (80 COCO classes) |
| `yolo26n.pt` | 2.57 M | 5.29 MB | Evaluated as a lighter alternative backbone |
| `pretrained_plate_detector.pt` | 3.01 M | 5.96 MB | Pre-fine-tuning checkpoint for the plate model |

## Environment these numbers came from

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB GDDR6, CUDA 12.4 |
| Precision | FP16 (Automatic Mixed Precision, `half=True`) |
| Concurrency | ThreadPoolExecutors — OCR: 4 workers, Helmet: 2, Gender: 2 |
| OS | Windows 11 (native) |

## How to reproduce / retrain

Training scripts are in `backend/`: `train_plate_model.py`,
`train_models.py` (helmet + gender), `prepare_data.py` (dataset prep),
`auto_annotate.py` (bootstrap labeling from existing weights). None of
the raw dataset images are committed to this repo (see `.gitignore`) —
you'll need your own source footage to retrain from scratch.

---

*Numbers sourced from the original project technical report
(May 2026). See [LIMITATIONS.md](LIMITATIONS.md) for the honest
assessment of what these metrics do and don't tell you.*
