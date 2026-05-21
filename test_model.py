from ultralytics import YOLO
import cv2
import os

def main():
    model_path = 'runs/detect/indian_plate_v1/weights/best.pt'
    video_path = 'videos/L1.mp4'
    output_path = 'test_plate_detection.jpg'
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please train your model first using: python3 train_plate_model.py")
        return
        
    if not os.path.exists(video_path):
        print(f"Error: Source video not found at {video_path}")
        return

    print(f"Loading custom model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    # Read frame 500 (good probability of containing vehicles/plates)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("Error: Could not read frame from video.")
        return
        
    print("Running license plate prediction...")
    results = model.predict(frame, conf=0.40)
    
    # Plot bounding boxes
    annotated = results[0].plot()
    
    cv2.imwrite(output_path, annotated)
    print(f"✅ Visual test complete! Saved prediction to: {output_path}")
    print("Open this image to visually inspect the plate detections.")

if __name__ == "__main__":
    main()
