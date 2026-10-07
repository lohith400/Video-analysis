# Pedestrian Detection & Demographic Model Research

## 1. Person Detection & Tracking (Counting)

### Current Problem
The existing pipeline extracted pedestrian candidates via `b.cls == 0` from `tracker.last_boxes`. Because `models/v2.0/vehicle_detector.pt` is an Indian vehicle detector whose class 0 is `Hatchback`, the tracker was tracking Hatchbacks as pedestrians. Moreover, rider exclusion used a flat IoU >= 0.30, failing to filter out riders and pillions whose bounding boxes only partially intersect the two-wheeler.

### Evaluated Detection Backbones
| Detector | Source | License | Model Size | CPU Speed (640p) | GPU Speed (960p) | Decision & Rationale |
|---|---|---|---|---|---|---|
| **YOLOv8n / YOLO11n (COCO, class 0)** | Ultralytics / GitHub | AGPL-3.0 | 6.2 MB | 18 ms | 2.8 ms | **PRIMARY CHOSEN**: High accuracy on full-body pedestrians, native integration, lightweight memory footprint (< 30 MB RAM). |
| **YOLOv8s / YOLO11s** | Ultralytics | AGPL-3.0 | 22.5 MB | 48 ms | 4.2 ms | Rejected: Higher CPU latency on Render free-tier. |
| **SAHI Tiled Inference** | SAHI GitHub | Apache 2.0 | N/A (wrapper) | 120+ ms | 15 ms | Rejected for real-time video: Multi-tile slicing is too slow for 15+ FPS real-time processing. |

### Tracking Strategy
- **Tracker**: ByteTrack with custom low-threshold second association (`track_high_thresh: 0.4`, `track_low_thresh: 0.1`, `new_track_thresh: 0.5`, `track_buffer: 30`).
- **Persistence Gate**: `MIN_PEDESTRIAN_TRACK_AGE = 8` frames before a pedestrian track is counted into cumulative statistics.
- **Rider Exclusion**: 
  - Overlap check: `intersection(person, vehicle) / person_area >= 0.50` (or >= 0.40 for two-wheelers/autos).
  - Spatial containment check: person center inside vehicle bounding box expanded vertically by 40% (to encapsulate seated riders and pillions).

---

## 2. Demographic Classification (Gender & Age)

### Evaluated Model Families
| Model Family | Specific Model / Checkpoint | Source URL | License | Size | Speed on CPU | Selected? | Rationale |
|---|---|---|---|---|---|---|---|
| **Pedestrian Attribute Recognition (PAR)** | PP-LCNet PAR / OpenPAR (PA-100K & RAP) | [PaddleClas / OpenPAR](https://github.com/Event-AHU/OpenPAR) | Apache 2.0 | 6.8 MB (ONNX) | ~6 ms / crop | **PRIMARY CHOICE**: Trained on 100K+ full-body CCTV pedestrian crops. Predicts `Female` (0/1) and `AgeLess18` (0/1) directly without relying on facial features. Extremely lightweight on CPU and memory (< 40 MB). |
| **Ultralytics YOLOv8n-cls Demographics** | YOLOv8n-cls (3-class: `male_adult`, `female_adult`, `child`) | [Ultralytics Releases](https://github.com/ultralytics/assets/releases) | AGPL-3.0 | 6.0 MB | ~5 ms / crop | **SECONDARY / FALLBACK CHOICE**: Fast native PyTorch inference through Ultralytics interface, 3-way softmax output, easily swappable via `config.py`. |
| **Vision-Language Zero-Shot (CLIP)** | MobileCLIP-S0 / CLIP ViT-B/32 | [Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32) | MIT | 45 MB – 350 MB | 35 – 110 ms | Rejected: Large memory footprint (> 400 MB RAM) risks OOM on Render 512 MB container; prompts on blurry CCTV crops have noisy zero-shot accuracy. |
| **Facial Demographic Estimators** | InsightFace / FairFace / DeepFace | [InsightFace](https://github.com/deepinsight/insightface) | Non-Commercial / MIT | 35 MB – 140 MB | 45 – 90 ms | Rejected as primary: CCTV traffic cameras view pedestrians from high angles, back/side angles, or at distances where faces are < 15x15 px or completely obscured. |
| **Binary Generic Detector** | Legacy `models/gender_detector.pt` | Local repo | Unknown | 6.2 MB | 6 ms | Discontinued: Inspecting metadata reveals only 2 classes (`female`, `male`) with zero child class and trained on an 84-sample dataset containing 100% class 0. |

---

## 3. Architecture & Modular Interface

All demographic inference is isolated behind a clean classifier interface:
```python
def classify(crop: np.ndarray) -> Tuple[str, float]:
    """
    Returns:
        label: 'male', 'female', 'child', or 'unknown'
        confidence: float in [0.0, 1.0]
    """
```

### Fallback & Robustness Strategy
1. **Primary Model**: Full-body CCTV PAR / Demographics classifier loaded from `models/v2.1/pedestrian_demographics.pt` (or ONNX).
2. **Quality Gates**: Reject crops with `height < MIN_CROP_HEIGHT` (80px), `width < MIN_CROP_WIDTH` (30px), Laplacian variance < `MIN_SHARPNESS` (20.0), or cut off at frame borders.
3. **Temporal Voting**: Per-track rolling history of last `VOTE_WINDOW` (15) predictions. Winner must have >= 60% weighted confidence and >= 3 votes.
4. **Honest Abstention**: If confidence < threshold or quality gate fails, return `unknown`. Never silently coerce to male.
5. **No Silent Failures**: If model file is missing at startup, print explicit `ERROR: Pedestrian demographic model missing at ...` and log active fallback mode.
