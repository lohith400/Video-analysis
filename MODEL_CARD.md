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
| Parameters | 3.01 M — 23.42 MB weights |
| Training run | `plate_v1`, 29/80 epochs (early stopped after convergence), batch 16, 960×960 input |
| Training time | ~17 min on RTX 3050 (29 epochs × ~35s/epoch) |
| **mAP@50** | **99.37%** |
| **mAP@50–95** | **83.09%** |
| **Precision / Recall** | **98.52% / 98.14%** |
| Inference speed | 8.3 ms/image (~120 FPS) on RTX 3050 |
| Dataset | 2,021 annotated images from Kaggle (`kedarsai/indian-license-plates-with-labels`) |
| Dataset split | 1,616 train / 202 val / 203 test (80/10/10, seed=42) |
| Skipped images | 62 unannotated background images excluded |
| Loss weights | Box=7.5 (elevated for small-object accuracy), Cls=0.5 |
| Early stopping | patience=20 — best weights auto-saved at peak mAP |
| Real-video test | 28/30 sampled frames detected plates on actual Indian traffic footage |

The 960×960 input resolution was chosen specifically because license plates
are small objects in wide traffic frames. The elevated box loss gain (7.5
vs default 5.0) improved tight bounding-box localization, closing the
mAP@50 → mAP@50-95 gap significantly compared to the previous 640×640
training run. The model is deployed as the production plate detector for
the ANPR pipeline.


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

## 3. Pedestrian Detector & Demographics Vision Engine (v2.1)

| Property | Value |
|---|---|
| Pedestrian Detector | YOLOv8-Nano (COCO class 0 `person`) at native resolution |
| Demographic Classifier | `models/v2.1/gender_detector.pt` behind modular `DemographicClassifier` interface |
| Classes | `male_adult`, `female_adult`, `child`, `unknown` |
| Tracking & Persistence | ByteTrack tracking with track age gate (`MIN_PEDESTRIAN_TRACK_AGE = 8`) |
| Rider Exclusion | Dual geometric gate: `intersection/person_area >= 0.50` + 40% upward-expanded two-wheeler box containment |
| Temporal Voting | Confidence-weighted rolling consensus (15-frame window, >=60% weighted majority, >=3 votes) |
| Child Logic | Model prediction priority (`USE_HEIGHT_CHILD_HEURISTIC = False` by default; weak height heuristic disabled) |
| Quality Gates | Min height 80px, min width 30px, Laplacian sharpness variance >= 20.0, frame edge cutoff rejection |
| Abstention Policy | Low confidence (<0.50) or gated crops honestly abstain as `unknown`; never coerced to male |
| Privacy Notice | Full-body demographic inference from public CCTV is inherently approximate and sensitive; `unknown` is actively logged when uncertainty exists |

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

*Plate model numbers sourced from the August 2026 retraining run
(see `Notes/license_plate_model_report.md`). Helmet and gender model
numbers sourced from the original project technical report (May 2026).
See [LIMITATIONS.md](LIMITATIONS.md) for the honest assessment of what
these metrics do and don't tell you.*

