from detector import VehicleModelLoader, crop_vehicle
from tracker import VehicleTracker
from ocr_engine import OCREngine
import cv2
import config
import torch

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    # 1. Load vehicle detector
    loader = VehicleModelLoader(device)
    tracker = VehicleTracker(loader.yolo, device, loader.vehicle_class_ids)
    
    # 2. Load OCR Engine (uses PlateDetector + EasyOCR)
    ocr = OCREngine(device)
    
    video_path = "uploads/L2.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    print("Processing video frames...")
    frame_idx = 0
    while cap.isOpened() and frame_idx < 60:  # process 2 seconds of video
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run vehicle detector
        results = tracker.model.predict(
            frame,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=loader.vehicle_class_ids,
            device=tracker.device,
            verbose=False
        )
        
        # Track vehicles
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            # Submit to OCR if needed
            for i, box in enumerate(xyxy):
                cls_name = tracker.model.names[int(cls[i])]
                if cls_name in config.PLATE_DETECTION_CLASSES:
                    crop = crop_vehicle(frame, box.astype(int))
                    if crop.size > 0:
                        track_id = frame_idx * 100 + i  # Unique track ID per detection for testing
                        if ocr.needs_ocr(track_id):
                            print(f"Submitting crop for dummy track {track_id} ({cls_name})...")
                            ocr.submit_vehicle_crop(track_id, crop, box.astype(int))
        
        ocr.drain_completed()
        frame_idx += 1
        
    print("Draining remaining OCR tasks...")
    # Wait a few seconds to let easyocr tasks complete in the background thread pool
    import time
    for _ in range(5):
        time.sleep(1)
        ocr.drain_completed()
        
    plates = ocr.get_all_plates()
    print("Detected plates:", plates)
    print("Total plates:", ocr.total_plates_detected)
    
    ocr.shutdown()
    cap.release()

if __name__ == "__main__":
    test()
