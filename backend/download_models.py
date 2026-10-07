#!/usr/bin/env python3
"""Download model weights; ensure models directory exists.

Vehicle model (yolov8n.pt) auto-downloads via the ultralytics library.
Custom-trained models (plate, helmet, gender) are fetched from the
v1.0-models GitHub Release if not already present locally.
"""

import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

from ultralytics import YOLO

import config

MODELS_DIR = Path("models")

# GitHub Release URLs for custom-trained weights
CUSTOM_MODEL_URLS = {
    config.PLATE_MODEL: "https://github.com/lohith400/Video-analysis/releases/download/v1.0-models/license_plate_detector.pt",
    config.HELMET_MODEL: "https://github.com/lohith400/Video-analysis/releases/download/v1.0-models/helmet_detector.pt",
    config.GENDER_MODEL: "https://github.com/lohith400/Video-analysis/releases/download/v1.0-models/gender_detector.pt",
}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Path("models/v2.0").mkdir(parents=True, exist_ok=True)
    Path("models/v2.1").mkdir(parents=True, exist_ok=True)

    # --- Pedestrian & vehicle base detector (yolov8n.pt via ultralytics) ---
    pedestrian_dest = Path(config.PEDESTRIAN_MODEL)
    print(f"Ensuring pedestrian detector at {pedestrian_dest} ...")
    try:
        model = YOLO("yolov8n.pt")
        if not pedestrian_dest.exists():
            src = Path(getattr(model, "ckpt_path", None) or "yolov8n.pt")
            if src.exists():
                shutil.copy2(src, pedestrian_dest)
            else:
                model.save(str(pedestrian_dest))
        print(f"  Pedestrian detector ready at {pedestrian_dest}")
    except Exception as exc:
        print(f"ERROR: Failed to load/download pedestrian detector model ({exc})", file=sys.stderr)
        sys.exit(1)

    # --- Vehicle model ---
    vehicle_dest = Path(config.VEHICLE_MODEL)
    if not vehicle_dest.exists():
        print(f"Copying vehicle model to {vehicle_dest} ...")
        src = Path("yolov8n.pt")
        if src.exists():
            shutil.copy2(src, vehicle_dest)

    # --- Custom-trained models (from GitHub Release or local fallback) ---
    for local_path, url in CUSTOM_MODEL_URLS.items():
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"Already present: {dest}")
            continue

        # Check local legacy location for gender model
        if "gender_detector.pt" in str(dest):
            legacy = Path("models/gender_detector.pt")
            if legacy.exists():
                shutil.copy2(legacy, dest)
                print(f"  Copied from local legacy path to {dest}")
                continue

        print(f"Downloading {dest.name} from release URL ...")
        try:
            urlretrieve(url, str(dest))
            print(f"  Saved: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  ERROR: Failed to download {dest}: {e}", file=sys.stderr)
            if dest == Path(config.GENDER_MODEL) or dest == Path(config.PEDESTRIAN_MODEL):
                sys.exit(1)

    print("All models ready.")


if __name__ == "__main__":
    main()
