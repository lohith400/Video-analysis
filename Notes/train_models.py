#!/usr/bin/env python3
"""
train_models.py
Indian Road Intelligence System (IRIS) — Machine Learning Script

Orchestrates custom YOLOv8 model training, evaluation reports, visual tests, and system deployment.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Run: pip install torch")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics YOLOv8 is not installed. Run: pip install ultralytics")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Run: pip install opencv-python")
    sys.exit(1)

# ==============================================================================
# CONFIG SECTION
# ==============================================================================
VIDEO_DIR = "videos"
HELMET_TRAIN_DIR = "helmet_dataset_train"
GENDER_TRAIN_DIR = "gender_dataset_train"
PLATE_TRAIN_DIR = "plate_dataset_train"
MODELS_DIR = "models"
TEST_OUTPUT_DIR = "test_outputs"
# ==============================================================================


def check_cuda() -> str:
    print("\nChecking hardware acceleration status...")
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        # Approximate VRAM calculation
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("  ● CUDA GPU acceleration: ACTIVE ✓")
        print(f"  ● GPU Name: {device_name}")
        print(f"  ● Total VRAM: {total_vram:.2f} GB")
        return "0"
    else:
        print("  ⚠ WARNING: CUDA GPU acceleration is NOT available.")
        print("  Training on CPU will be extremely slow.")
        response = input("  Do you want to continue training on CPU? (y/n): ").strip().lower()
        if response != 'y':
            print("  Training aborted.")
            sys.exit(0)
        return "cpu"


def train_model(mode: str) -> None:
    device = check_cuda()

    if mode == "helmet":
        train_root = Path(HELMET_TRAIN_DIR)
        name = "helmet_v1"
        box_gain = 7.5
    elif mode == "plate":
        train_root = Path(PLATE_TRAIN_DIR)
        name = "plate_v1"
        box_gain = 7.5  # plates are small objects — weight box loss heavily
    else:
        train_root = Path(GENDER_TRAIN_DIR)
        name = "gender_v1"
        box_gain = 5.0

    yaml_file = train_root / "data.yaml"
    img_dir = train_root / "train" / "images"

    # Pre-train checks
    if not yaml_file.exists():
        print(f"\nError: Dataset configuration file not found at '{yaml_file}'")
        if mode == "plate":
            print("Please run: python prepare_plate_data.py --source <your_raw_dataset_folder>")
        else:
            print(f"Please run: python prepare_data.py --step setup_{mode}")
        sys.exit(1)

    if not img_dir.exists() or not any(img_dir.iterdir()):
        print(f"\nError: Training images not found inside '{img_dir}'")
        if mode == "plate":
            print("Please make sure your dataset was prepared using: python prepare_plate_data.py --source <your_raw_dataset_folder>")
        else:
            print(f"Please make sure split files are populated correctly using: python prepare_data.py --step setup_{mode}")
        sys.exit(1)

    print(f"\nInitializing YOLOv8 Nano base model for {mode} training...")
    model = YOLO("yolov8n.pt")

    # Plates are small objects relative to the frame — a larger input
    # resolution noticeably helps detection recall on distant/angled plates.
    imgsz = 960 if mode == "plate" else 640
    epochs = 80 if mode == "plate" else 60

    print(f"\nStarting YOLOv8 training on {train_root.name} (name={name}, device={device})...")
    
    # Run YOLOv8 training
    try:
        model.train(
            data=str(yaml_file),
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            name=name,
            device=0 if device == "0" else "cpu",
            half=True,
            patience=20,
            save=True,
            val=True,
            plots=True,
            box=box_gain,
            cls=0.5,
        )
    except Exception as exc:
        print(f"\nTraining crashed with error: {exc}")
        sys.exit(1)

    best_path = Path("runs") / "detect" / name / "weights" / "best.pt"
    print("\n=======================================================")
    print("TRAINING COMPLETED SUCCESSFUL ✓")
    print("=======================================================")
    if best_path.exists():
        print(f"Best model weights saved to: {best_path}")
    else:
        print("Trained model weights saved inside runs/detect/ folder.")
    print("=======================================================")
    print("Next step, run:")
    print(f"  python train_models.py --step evaluate_{mode}")
    print("=======================================================")


def find_best_weights(name: str) -> Path:
    # First priority: runs/detect/name/weights/best.pt
    default_best = Path("runs") / "detect" / name / "weights" / "best.pt"
    if default_best.exists():
        return default_best

    # Backup priority: search runs/detect for any folder starting with the prefix name
    detect_dir = Path("runs") / "detect"
    if not detect_dir.exists():
        raise FileNotFoundError(f"Error: YOLO runs folder '{detect_dir}' does not exist.")

    matching_runs = [d for d in detect_dir.iterdir() if d.is_dir() and d.name.startswith(name)]
    if not matching_runs:
        raise FileNotFoundError(f"Error: No trained weight folders found starting with name '{name}' under '{detect_dir}'.")

    # Take the most recently modified run folder
    matching_runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for run in matching_runs:
        best_pt = run / "weights" / "best.pt"
        if best_pt.exists():
            return best_pt

    raise FileNotFoundError(f"Error: Could not find best.pt file under most recent weights: {matching_runs[0]}")


def evaluate_model(mode: str) -> None:
    if mode == "helmet":
        yaml_file = Path(HELMET_TRAIN_DIR) / "data.yaml"
        run_name = "helmet_v1"
        classes = ["rider_helmet", "rider_no_helmet", "pillion_helmet", "pillion_no_helmet"]
    elif mode == "plate":
        yaml_file = Path(PLATE_TRAIN_DIR) / "data.yaml"
        run_name = "plate_v1"
        classes = ["license_plate"]
    else:
        yaml_file = Path(GENDER_TRAIN_DIR) / "data.yaml"
        run_name = "gender_v1"
        classes = ["male_adult", "female_adult", "child"]

    if not yaml_file.exists():
        print(f"\nError: Dataset configuration file not found at '{yaml_file}'")
        sys.exit(1)

    print(f"\nScanning for most recently trained {mode} weights...")
    try:
        best_pt = find_best_weights(run_name)
        print(f"Found best model weights at: {best_pt}")
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        print(f"Make sure you ran the training step: python train_models.py --step train_{mode}")
        sys.exit(1)

    print("\nLoading model and executing validation on dataset splits...")
    model = YOLO(str(best_pt))
    
    # Run validation checks
    val_results = model.val(data=str(yaml_file), plots=True, verbose=False)

    # Retrieve overall metrics
    # Note: val_results has maps metric attributes:
    # val_results.results_dict contains 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    rdict = val_results.results_dict
    mAP50 = rdict.get("metrics/mAP50(B)", 0.0)
    mAP50_95 = rdict.get("metrics/mAP50-95(B)", 0.0)
    precision = rdict.get("metrics/precision(B)", 0.0)
    recall = rdict.get("metrics/recall(B)", 0.0)

    # Print gorgeous report block
    print("\n================================================")
    print(f"{mode.upper()} MODEL EVALUATION RESULTS")
    print("================================================")
    print(f"Overall mAP50:       {mAP50:.4f}")
    print(f"Overall mAP50-95:    {mAP50_95:.4f}")
    print(f"Overall Precision:   {precision:.4f}")
    print(f"Overall Recall:      {recall:.4f}")
    print("------------------------------------------------")
    print("Per Class Results:")
    
    # Retrieve per class AP values
    # val_results.box.ap contains list of AP50-95 per class, val_results.box.ap50 contains AP50 per class
    ap50_vals = val_results.box.ap50
    for idx, name in enumerate(classes):
        ap = ap50_vals[idx] if idx < len(ap50_vals) else 0.0
        print(f"  {name:<20} AP = {ap:.4f}")
        
    print("------------------------------------------------")
    print("Research Paper Values:")
    print(f"  mAP50 = {mAP50:.4f}  (report this in your paper)")
    print("================================================\n")

    # Evaluation guidance
    if mAP50 > 0.80:
        print("Guidance: Excellent. Run deploy step.")
    elif mAP50 > 0.70:
        print("Guidance: Good. Acceptable for research.")
    else:
        print("Guidance: Need improvement. Annotate 200 more images and retrain.")

    # Visual inference test
    run_visual_test(model, mode)


def run_visual_test(model: YOLO, mode: str) -> None:
    print(f"\nInitiating visual inference check for {mode}...")
    video_dir_path = Path(VIDEO_DIR)
    
    # Find any mp4 video to grab a frame
    video_file = video_dir_path / "L1.mp4"
    if not video_file.exists():
        # Fallback search
        mp4_files = sorted([p for p in video_dir_path.iterdir() if p.suffix.lower() == ".mp4"])
        if mp4_files:
            video_file = mp4_files[0]
        else:
            print(f"Warning: No video files (.mp4) found under '{VIDEO_DIR}/' to extract a visual test frame.")
            return

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Warning: Could not open {video_file.name} for visual frame test.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Warning: Failed to capture test frame from video.")
        return

    print(f"Captured test frame from video: {video_file.name}")
    
    # Run prediction
    results = model.predict(frame, conf=0.4, verbose=False)
    
    if results and len(results) > 0:
        annotated_frame = results[0].plot()
        test_out_dir = Path(TEST_OUTPUT_DIR)
        test_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = test_out_dir / f"{mode}_test.jpg"
        cv2.imwrite(str(out_path), annotated_frame)
        print("Visual test image saved successfully ✓")
        print(f"Visual test path: {out_path}")
        print("=======================================================")
        print("Next step, run:")
        print("  python train_models.py --step deploy")
        print("=======================================================")


def deploy_models() -> None:
    print("\nStarting system deployment of trained weights...")
    models_path = Path(MODELS_DIR)
    models_path.mkdir(parents=True, exist_ok=True)

    deployed_status = {}

    # 1. Deploy Helmet Detector
    try:
        helmet_best = find_best_weights("helmet_v1")
        helmet_dest = models_path / "helmet_detector.pt"
        shutil.copy2(helmet_best, helmet_dest)
        deployed_status["helmet"] = {
            "source": helmet_best,
            "dest": helmet_dest,
            "size": helmet_dest.stat().st_size / (1024 * 1024),
            "expected_classes": 4
        }
        print(f"  ✓ Deployed helmet model successfully: {helmet_dest.name}")
    except FileNotFoundError as exc:
        print(f"  ⚠ WARNING: Helmet model deployment skipped: {exc}")
    except Exception as exc:
        print(f"  ⚠ ERROR: Failed to deploy helmet model: {exc}")

    # 2. Deploy Gender Detector
    try:
        gender_best = find_best_weights("gender_v1")
        gender_dest = models_path / "gender_detector.pt"
        shutil.copy2(gender_best, gender_dest)
        deployed_status["gender"] = {
            "source": gender_best,
            "dest": gender_dest,
            "size": gender_dest.stat().st_size / (1024 * 1024),
            "expected_classes": 3
        }
        print(f"  ✓ Deployed gender model successfully: {gender_dest.name}")
    except FileNotFoundError as exc:
        print(f"  ⚠ WARNING: Gender model deployment skipped: {exc}")
    except Exception as exc:
        print(f"  ⚠ ERROR: Failed to deploy gender model: {exc}")

    # 3. Deploy Plate Detector
    try:
        plate_best = find_best_weights("plate_v1")
        plate_dest = models_path / "license_plate_detector.pt"
        shutil.copy2(plate_best, plate_dest)
        deployed_status["plate"] = {
            "source": plate_best,
            "dest": plate_dest,
            "size": plate_dest.stat().st_size / (1024 * 1024),
            "expected_classes": 1
        }
        print(f"  ✓ Deployed plate model successfully: {plate_dest.name}")
    except FileNotFoundError as exc:
        print(f"  ⚠ WARNING: Plate model deployment skipped: {exc}")
    except Exception as exc:
        print(f"  ⚠ ERROR: Failed to deploy plate model: {exc}")

    if not deployed_status:
        print("\nError: No models were deployed. Make sure you trained at least one model first.")
        sys.exit(1)

    print("\nRunning quick load verification tests...")
    
    # Verify and load models
    for key, info in deployed_status.items():
        try:
            model = YOLO(str(info["dest"]))
            names = list(model.names.values())
            info["classes"] = names
            print(f"  ● Loaded {info['dest'].name} ✓  Classes found ({len(names)}): {names}")
        except Exception as exc:
            print(f"  ⚠ ERROR: Failed to verify loading model {info['dest'].name}: {exc}")

    print("\n=======================================================")
    print("DEPLOYMENT COMPLETE")
    print("=======================================================")
    
    for key, info in deployed_status.items():
        print(f"  {info['dest'].name:<20} → {info['size']:.2f} MB  Classes: {len(info.get('classes', []))}")

    print("\nBoth models are ready in your models/ directory.")
    print("Your IRIS Road Intelligence system will automatically activate them at startup!")
    print("=======================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IRIS Road Intelligence System — Machine Learning Scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Steps:
  train_helmet      Validates setup, runs YOLOv8 custom helmet training on GPU.
  train_gender      Validates setup, runs YOLOv8 custom gender/child training on GPU.
  train_plate       Validates setup, runs YOLOv8 custom license-plate training on GPU.
  evaluate_helmet   Validates metrics, prints mAP metrics table, and saves visual test.
  evaluate_gender   Validates metrics, prints mAP metrics table, and saves visual test.
  evaluate_plate    Validates metrics, prints mAP metrics table, and saves visual test.
  deploy            Copies models to models/ folder and verifies sizes and classes.

Plate training requires prepare_plate_data.py to be run first:
  python prepare_plate_data.py --source <your_raw_dataset_folder>
  python train_models.py --step train_plate
  python train_models.py --step evaluate_plate
  python train_models.py --step deploy
        """
    )
    parser.add_argument(
        "--step",
        choices=[
            "train_helmet", "train_gender", "train_plate",
            "evaluate_helmet", "evaluate_gender", "evaluate_plate",
            "deploy",
        ],
        help="Specify the machine learning step to execute."
    )

    args = parser.parse_args()

    if not args.step:
        parser.print_help()
        sys.exit(0)

    if args.step == "train_helmet":
        train_model("helmet")
    elif args.step == "train_gender":
        train_model("gender")
    elif args.step == "train_plate":
        train_model("plate")
    elif args.step == "evaluate_helmet":
        evaluate_model("helmet")
    elif args.step == "evaluate_gender":
        evaluate_model("gender")
    elif args.step == "evaluate_plate":
        evaluate_model("plate")
    elif args.step == "deploy":
        deploy_models()


if __name__ == "__main__":
    main()
