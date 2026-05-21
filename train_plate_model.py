from ultralytics import YOLO
import torch

def main():
    # 1. Verify GPU availability inside WSL2
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA is not available. Training will run on CPU and be VERY slow.")

    # 2. Load the base model (starts from COCO pretrained weights)
    print("Loading base YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # 3. Train using transfer learning on the license plate dataset
    print("Starting fine-tuning...")
    results = model.train(
        data     = 'plate_dataset/data.yaml',
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,                 # Reduce to 8 or 4 if you run out of VRAM (Out of Memory)
        name     = 'indian_plate_v1',
        device   = 0 if torch.cuda.is_available() else 'cpu',
        amp      = True,               # Correct way to enable fast FP16 mixed precision training (replaces half=True)
        patience = 15,                 # Stops early if mAP stops improving
        save     = True,
        val      = True,
        plots    = True,
        verbose  = True,
    )

    print("\n✅ Training complete!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
