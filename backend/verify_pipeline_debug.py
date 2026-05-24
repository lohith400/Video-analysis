from detector import VehicleModelLoader, PlateDetector, crop_vehicle
from tracker import VehicleTracker
import cv2
import config
import torch
import easyocr
import re

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    loader = VehicleModelLoader(device)
    detector = PlateDetector(device)
    reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
    
    video_path = "uploads/L2.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    print("Scanning video for vehicles and plates...")
    frame_idx = 0
    plates_found = 0
    
    while cap.isOpened() and frame_idx < 120:  # Scan up to 4 seconds
        ret, frame = cap.read()
        if not ret:
            break
            
        results = loader.yolo.predict(
            frame,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=loader.vehicle_class_ids,
            device=device,
            verbose=False
        )
        
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            for i, box in enumerate(xyxy):
                cls_name = loader.yolo.names[int(cls[i])]
                if cls_name in config.PLATE_DETECTION_CLASSES:
                    crop = crop_vehicle(frame, box.astype(int))
                    if crop.size > 0:
                        # Try to detect plate
                        plate_results = detector.model.predict(
                            crop,
                            conf=0.1,  # Try a very low threshold for debugging!
                            iou=config.IOU_THRESHOLD,
                            device=device,
                            verbose=False
                        )
                        
                        pboxes = plate_results[0].boxes
                        if pboxes is not None and len(pboxes) > 0:
                            best_idx = int(pboxes.conf.argmax().item())
                            conf = float(pboxes.conf[best_idx].item())
                            px1, py1, px2, py2 = pboxes.xyxy[best_idx].cpu().numpy().astype(int)
                            print(f"[Frame {frame_idx}] Found vehicle ({cls_name}). BBox plate conf: {conf:.4f}")
                            
                            # Extract plate crop
                            h, w = crop.shape[:2]
                            px1, py1 = max(0, px1), max(0, py1)
                            px2, py2 = min(w, px2), min(h, py2)
                            plate_crop = crop[py1:py2, px1:px2]
                            
                            if plate_crop.size > 0:
                                ocr_res = reader.readtext(plate_crop)
                                raw_text = "".join([res[1] for res in ocr_res])
                                clean_text = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
                                print(f"  -> Plate Crop Size: {plate_crop.shape}. Raw OCR: '{raw_text}', Clean: '{clean_text}' (len={len(clean_text)})")
                                if len(clean_text) >= 4:
                                    plates_found += 1
        frame_idx += 1
        
    print(f"\nScan complete! Total plates found with len >= 4: {plates_found}")
    cap.release()

if __name__ == "__main__":
    test()
