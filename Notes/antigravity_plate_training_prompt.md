# Prompt for Antigravity — IRIS License Plate Model: Dataset + Training Pipeline

Copy everything below this line into Antigravity as your task/prompt.

---

## Context

I'm building **IRIS (Indian Road Intelligence System)**, a traffic video analysis
pipeline. Repo root contains a `backend/` folder with an existing FastAPI + YOLOv8
codebase. I already have two working Python scripts inside `backend/` that you must
use exactly as they are, without modifying their logic unless a real bug blocks
execution:

- `backend/prepare_plate_data.py` — converts a raw license-plate dataset (Pascal VOC
  XML annotations OR pre-existing YOLO `.txt` annotations) into a YOLOv8-ready
  train/val/test split with a `data.yaml` file, saved under
  `backend/plate_dataset_train/`.
- `backend/train_models.py` — the project's existing ML orchestration script. It now
  supports a `plate` mode alongside its existing `helmet` and `gender` modes, via
  these CLI steps:
  - `python train_models.py --step train_plate`
  - `python train_models.py --step evaluate_plate`
  - `python train_models.py --step deploy`

Your job is to execute the **full pipeline end to end**: download the dataset,
prepare it, train the model, evaluate it, and deploy the trained weights into the
project — following the exact steps below, in the exact order given. Do not skip
steps. Do not improvise alternate approaches unless a step fails and you need to
troubleshoot — in that case, explain what failed and why before trying a fix.

---

## Step 0 — Environment setup

Run these first, inside the `backend/` folder of the repo:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install ultralytics torch opencv-python Pillow tqdm kaggle
```

Confirm GPU availability (not required, but training will be much faster with one):

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, that's fine — `train_models.py` already
detects this and will prompt to continue on CPU (training will just take longer).

---

## Step 1 — Get Kaggle API credentials

The dataset is hosted on Kaggle and requires an API token to download via CLI.

1. Go to https://www.kaggle.com/settings
2. Scroll to the "API" section and click "Create New Token"
3. This downloads a file called `kaggle.json`
4. Place it here:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
5. Set correct permissions (Linux/Mac only):
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Verify it works before proceeding:**
```bash
kaggle datasets list -s "indian license plates"
```
You should see `kedarsai/indian-license-plates-with-labels` in the results. If this
command fails with an auth error, stop and fix the `kaggle.json` placement before
continuing — do not proceed to Step 2 until this succeeds.

---

## Step 2 — Download the dataset

From the repo root (one level above `backend/`):

```bash
kaggle datasets download -d kedarsai/indian-license-plates-with-labels
unzip indian-license-plates-with-labels.zip -d plate_dataset_raw
```

After this, inspect the folder structure:

```bash
find plate_dataset_raw -maxdepth 3 | head -50
```

**Important — check the annotation format before moving on.** Look for either:
- `.xml` files (Pascal VOC format) sitting next to `.jpg`/`.png` images, or in a
  subfolder named `annotations/`
- `.txt` files (YOLO format) sitting next to images, or in a subfolder named
  `labels/`

`prepare_plate_data.py` auto-detects both formats using this exact search logic (do
not second-guess this — it already checks all these locations):
```
<image>.xml
<image>.txt
<parent_of_images>/annotations/<image_stem>.xml
<parent_of_images>/labels/<image_stem>.txt
<source_root>/annotations/<image_stem>.xml
<source_root>/labels/<image_stem>.txt
```

If the downloaded dataset uses a *different* structure than this (e.g. a single CSV
file with all bounding boxes instead of per-image XML/txt files), **stop and report
this back before proceeding** — the conversion script will silently skip every image
with no matching label file, and you'll end up training on zero data without an
obvious error.

---

## Step 3 — Convert to YOLOv8 format

From inside `backend/`:

```bash
cd backend
python prepare_plate_data.py --source ../plate_dataset_raw
```

Expected output ends with:
```
=======================================================
PLATE DATASET PREPARATION COMPLETE ✓
=======================================================
Total usable images: <some number > 0>
Next step, run:
  python train_models.py --step train_plate
=======================================================
```

**Validation checks — run these before proceeding to training:**

```bash
cat plate_dataset_train/data.yaml
```
Expected content:
```yaml
path: <absolute path>/backend/plate_dataset_train
train: train/images
val: val/images
test: test/images
nc: 1
names: ['license_plate']
```

```bash
find plate_dataset_train -name "*.jpg" -o -name "*.png" | wc -l
find plate_dataset_train -name "*.txt" -path "*/labels/*" | wc -l
```
These two counts should be close to equal (every image should have a matching label
file). If the label count is dramatically lower than the image count, or if
"Converted" in the script's printed summary was 0, **stop and report back** — do not
proceed to train on a broken or empty dataset.

**If the "Skipped" count printed by the script was high** (many images had no
detected label file), open one raw annotation file manually and paste its exact
content into your report to me — the format might differ from standard VOC XML and
the converter's `voc_xml_to_yolo` function may need a small adjustment to match the
real tag names used (e.g. some datasets use `<box>` instead of `<bndbox>`, or store
class name as `<n>` instead of `<name>`).

---

## Step 4 — Optional: merge in a second, larger generic dataset

If you also have access to a large generic (non-India-specific) plate dataset — for
example a Roboflow export with ~24,000 images — you can merge it into the same
training pool to give the model more general "what does a plate look like" exposure
before it specializes on Indian plates. Run the exact same command again, pointing at
the second dataset's folder:

```bash
python prepare_plate_data.py --source ../roboflow_plate_dataset_raw
```

This is additive — it will not overwrite the first dataset's converted images,
because the script prefixes every output filename with its source folder name to
avoid collisions. Re-run the Step 3 validation checks afterward to confirm the total
image count increased as expected.

Skip this step entirely if you don't have a second dataset ready — it is optional and
the pipeline works fine with just the Indian dataset alone.

---

## Step 5 — Train the model

From inside `backend/`:

```bash
python train_models.py --step train_plate
```

This will:
- Print CUDA/GPU status and (if no GPU) ask you to confirm CPU training — answer `y`
  to continue
- Load `yolov8n.pt` as the base model (downloads automatically on first run if not
  cached)
- Train for 80 epochs at image size 960×960 (deliberately higher resolution than the
  helmet/gender models, since license plates are small objects in a full traffic
  frame and need finer resolution to detect reliably)
- Save results under `runs/detect/plate_v1/`

This step can take anywhere from 20 minutes (modern GPU) to several hours (CPU only)
depending on dataset size and hardware. Let it run to completion — do not interrupt
it partway through.

Expected final output:
```
=======================================================
TRAINING COMPLETED SUCCESSFUL ✓
=======================================================
Best model weights saved to: runs/detect/plate_v1/weights/best.pt
=======================================================
Next step, run:
  python train_models.py --step evaluate_plate
=======================================================
```

---

## Step 6 — Evaluate the model

```bash
python train_models.py --step evaluate_plate
```

This prints an mAP50 / mAP50-95 / precision / recall report, plus a per-class AP
breakdown (only one class here: `license_plate`), and saves a visual test image with
predicted bounding boxes drawn on a sample frame from `backend/videos/` (if any video
files exist there) to `backend/test_outputs/plate_test.jpg`.

**Report back to me:**
- The exact mAP50 value printed
- The exact precision and recall values
- Whether the script printed "Excellent", "Good", or "Need improvement" guidance

Do not proceed to deployment if mAP50 is very low (below roughly 0.5) — flag this
back to me first so we can decide whether to add more data, adjust training
hyperparameters, or debug the dataset before deploying a weak model.

---

## Step 7 — Deploy the trained weights

Only run this once Step 6's results look reasonable:

```bash
python train_models.py --step deploy
```

This copies `runs/detect/plate_v1/weights/best.pt` to
`backend/models/license_plate_detector.pt` — the exact path the main IRIS pipeline
(`server.py` / `detector.py`) already expects for the plate detector. It also
reloads the copied model and prints its detected class list to confirm it loaded
correctly.

**Note:** the deploy step will also attempt to deploy `helmet` and `gender` models if
their trained weights exist under `runs/detect/helmet_v1/` or `runs/detect/gender_v1/`
— if those haven't been trained yet, you'll see a `⚠ WARNING` for them, which is
expected and not an error. Only the plate deployment result matters for this task.

---

## Step 8 — Final verification

Confirm the deployed file exists and is a valid, loadable model:

```bash
ls -la backend/models/license_plate_detector.pt
python3 -c "
from ultralytics import YOLO
m = YOLO('backend/models/license_plate_detector.pt')
print('Classes:', m.names)
"
```

Expected output: `Classes: {0: 'license_plate'}`

Report this final confirmation back to me along with the mAP50 score from Step 6.
This completes the plate detection model training pipeline.

---

## Rules to follow throughout

- Do not modify `prepare_plate_data.py` or `train_models.py` unless a step genuinely
  fails due to a real bug (e.g. the annotation format doesn't match what the script
  expects) — in that case, explain exactly what failed before making any change.
- Do not skip the validation checks in Steps 3 and 6 — they exist specifically to
  catch silent failures (e.g. training on an empty or mislabeled dataset) before
  wasting time on a full training run.
- Do not delete `plate_dataset_raw/` or `plate_dataset_train/` after training
  finishes — I may want to inspect them or retrain with adjusted settings later.
- If any step's actual output doesn't match the "Expected output" shown above, stop
  and report the mismatch rather than assuming it's fine and continuing.
