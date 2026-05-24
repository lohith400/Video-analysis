import os
import glob
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def main():
    # 1. Configuration Paths
    repo_id = "keremberke/yolov8n-license-plate-detector"
    filename = "best.pt"
    images_dir = "dataset/images/all"
    labels_dir = "dataset/labels/all"
    
    os.makedirs("models", exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 2. Download the pre-trained weights using huggingface_hub (resolves LFS & redirects anonymously!)
    weights_path = "models/pretrained_plate_detector.pt"
    if not os.path.exists(weights_path):
        print("Downloading pre-trained license plate detector from Hugging Face Hub...")
        try:
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
            import shutil
            shutil.copy2(downloaded_path, weights_path)
            print("Successfully downloaded weights!")
        except Exception as e:
            print(f"Error downloading pre-trained weights via Hugging Face Hub: {e}")
            print("Attempting backup repository...")
            try:
                # Backup repo in case the main one fails
                downloaded_path = hf_hub_download(repo_id="koushim/yolov8-license-plate-detection", filename="best.pt")
                import shutil
                shutil.copy2(downloaded_path, weights_path)
                print("Successfully downloaded backup weights!")
            except Exception as e2:
                print(f"Backup download failed as well: {e2}")
                return
            
    # 3. Load the model
    print(f"Loading pre-trained license plate detector from {weights_path}...")
    model = YOLO(weights_path)
    
    # 4. Find images to annotate
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    print(f"Found {len(image_paths)} images to auto-annotate.")
    
    annotated_count = 0
    skipped_count = 0
    
    # 5. Run prediction and generate label files
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # Run inference
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            with open(label_path, "w") as f:
                for box in boxes:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    # Class ID is 0 for license_plate
                    xywhn = box.xywhn[0].tolist()
                    f.write(f"0 {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
            annotated_count += 1
        else:
            # If no plates detected, write an empty file to indicate a background image
            # (standard YOLO practices)
            with open(label_path, "w") as f:
                pass
            skipped_count += 1
            
    print(f"\n✅ Auto-annotation complete!")
    print(f"  - Annotated with plates: {annotated_count} images")
    print(f"  - Empty background frames: {skipped_count} images")
    print(f"All YOLO label files (.txt) saved successfully to: {labels_dir}/")

if __name__ == "__main__":
    main()
