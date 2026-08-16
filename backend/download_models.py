#!/usr/bin/env python3
"""Download model weights; ensure models directory exists.

Vehicle model (yolov8n.pt) auto-downloads via the ultralytics library.
Custom-trained models (plate, helmet, gender) are fetched from the
v1.0-models GitHub Release if not already present locally.
"""

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

    # --- Vehicle model (generic COCO, auto-downloads via ultralytics) ---
    vehicle_dest = Path(config.VEHICLE_MODEL)
    print(f"Downloading {config.VEHICLE_MODEL} ...")
    model = YOLO("yolov8n.pt")
    if not vehicle_dest.exists():
        import shutil

        src = Path(getattr(model, "ckpt_path", None) or "yolov8n.pt")
        if src.exists():
            shutil.copy2(src, vehicle_dest)
        else:
            model.save(str(vehicle_dest))

    # --- Custom-trained models (from GitHub Release) ---
    for local_path, url in CUSTOM_MODEL_URLS.items():
        dest = Path(local_path)
        if dest.exists():
            print(f"Already present: {dest}")
            continue
        print(f"Downloading {dest.name} from GitHub Release ...")
        try:
            urlretrieve(url, str(dest))
            print(f"  Saved: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  WARNING: Failed to download {dest.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
