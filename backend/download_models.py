#!/usr/bin/env python3
"""Download YOLOv8n weights; ensure models directory exists."""

from pathlib import Path

from ultralytics import YOLO

import config

MODELS_DIR = Path("models")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

    plate_dest = Path(config.PLATE_MODEL)
    if not plate_dest.exists():
        print(
            f"\nWARNING: {plate_dest} not found.\n"
            "Place your trained license_plate_detector.pt in models/\n"
            "Example: wget -O models/license_plate_detector.pt <your-url>\n"
        )
    else:
        print(f"Plate model present: {plate_dest}")

    print("Done.")


if __name__ == "__main__":
    main()
