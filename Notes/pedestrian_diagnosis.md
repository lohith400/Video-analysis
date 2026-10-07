# IRIS Pedestrian & Gender/Demographics System Diagnosis

## 1. Executive Summary
An in-depth audit of the IRIS pedestrian detection and demographic classification pipeline was conducted on branch `version2.1`. The diagnosis confirms all 10 root causes identified in the problem statement, with concrete code references, terminal inspections, and architectural analysis.

---

## 2. Model Structure & Inspection Evidence

Direct inspection using `ultralytics.YOLO` on active models:
- **Vehicle Model (`models/v2.0/vehicle_detector.pt`)**:
  - Task: `detect`
  - Classes: `{0: 'Hatchback', 1: 'Sedan', 2: 'SUV', 3: 'MUV', 4: 'Bus', 5: 'Truck', 6: 'Three-wheeler', 7: 'Two-wheeler', 8: 'LCV', 9: 'Mini-bus', 10: 'tempo-traveller', 11: 'bicycle', 12: 'Van', 13: 'Others'}`
  - **Critical Finding**: Class `0` is `'Hatchback'`. The vehicle model contains **ZERO** person classes.
- **Gender Model (`models/gender_detector.pt`)**:
  - Task: `classify`
  - Classes: `{0: 'female', 1: 'male'}`
  - **Critical Finding**: The model is a 2-class classifier only (`female`, `male`). It has no `child` class.

---

## 3. Root Cause Verification Matrix

| # | Hypothesized Root Cause | Status | Evidence & Concrete Finding |
|---|---|---|---|
| **1** | `backend/gender_dataset/labels/all/*.txt` only class `0` (male) | **CONFIRMED** | Evaluated all 84 label files: exactly 84 files, each containing class `0` (`{'0': 84}`). Zero samples for female or child in the dataset. |
| **2** | `config.py` `GENDER_MODEL = "models/gender_detector.pt"` path and class mismatch | **CONFIRMED** | Path is unversioned (`models/gender_detector.pt` vs `models/v2.0/...`). Model metadata is binary `{0: 'female', 1: 'male'}` with no `child` category. |
| **3** | `_run_classification` dual logic (detection vs classification) | **CONFIRMED** | `backend/pedestrian_detector.py` lines 139–173 checks both `probs` and `boxes`. Detection mode on crops falls back to `"unknown"` when detecting generic `"person"`. |
| **4** | Frame-to-frame label flip-flopping in `drain_completed` | **CONFIRMED** | `backend/pedestrian_detector.py` line 219: `self.results[tid] = gender` overwrites unconditionally on every frame. No per-track vote accumulator or temporal confidence history. |
| **5** | Naive scene child heuristic (`height < 0.60 * avg`) | **CONFIRMED** | `backend/pedestrian_detector.py` line 216 & `geometry_utils.py` line 40: perspective scaling causes distant adult pedestrians to have smaller pixel heights, falsely converting up to 50% of people to `child`. Overrides classifier output unconditionally. |
| **6** | Rider/pillion false-positive inclusion due to vehicle IoU | **CONFIRMED** | `backend/pedestrian_detector.py` line 94: `calculate_iou(box_a, v_box) >= config.PEDESTRIAN_VEHICLE_IOU` (0.30). A rider bbox is substantially smaller than a motorcycle/scooter/auto bbox, yielding an IoU of 0.10–0.25, bypassing exclusion. |
| **7** | `main.py` person source assumed `cls == 0` from vehicle model | **CONFIRMED** | `backend/main.py` line 245: `person_boxes = [b for b in raw_results if b.cls == 0]`. In `models/v2.0/vehicle_detector.pt`, class `0` is `'Hatchback'`. If auxiliary tracker fails or doesn't merge, all hatchbacks are treated as pedestrian crops. |
| **8** | Session total cumulative ID switches | **CONFIRMED** | `backend/main.py` lines 256–258: `_all_pedestrians_seen[p.track_id] = gender` records every track ID seen without a minimum track age or persistence filter. Temporary occlusions and ID swaps drastically inflate totals. |
| **9** | Instantaneous snapshots in `get_current_pedestrians` | **CONFIRMED** | `backend/pedestrian_detector.py` line 229: uses active frame bboxes without temporal smoothing or rolling median window, causing CSV snapshot spikes and drops. |
| **10** | Missing quality gates for crops | **CONFIRMED** | `backend/pedestrian_detector.py` line 121: accepts any crop where `size > 0`. Tiny bboxes (e.g., 20x40 px) and blurry crops are submitted for demographic classification. |

---

## 4. Additional Architectural Deficiencies Identified
11. **Silent Fallback Anti-Pattern**: `pedestrian_detector.py` lines 45–54 attempts to silently import `deepface` if model is missing, but `deepface` is not in `requirements.txt`.
12. **Inconsistent Crop Methods**: `main.py` used `crop_vehicle` on pedestrian boxes rather than person-specific padded cropping.
13. **Lack of Pedestrian Zone Gate**: No spatial verification or pedestrian ROI filtering to avoid detecting distant artifacts outside the road/sidewalk.
