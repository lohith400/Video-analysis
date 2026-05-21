import cv2
import os
import glob

# 1. Configuration
VIDEO_SOURCES = ['uploads/L2.mp4', 'uploads/L3.mp4']
OUTPUT_DIR    = 'dataset/images/all'
FRAME_SKIP    = 30   # Was 10. Set to 30 (1 frame per second) for maximum diversity and no redundancy!
MAX_FRAMES    = 800  # Cap overall target

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('dataset/labels/all', exist_ok=True)

# 2. Clear out the old repetitive images first to avoid mixing datasets
old_images = glob.glob(os.path.join(OUTPUT_DIR, '*.jpg'))
for f in old_images:
    try:
        os.remove(f)
    except OSError:
        pass
print(f"Cleared {len(old_images)} old repetitive frames.")

saved = 0

# 3. Extract frames from multiple video sources
for video_path in VIDEO_SOURCES:
    if not os.path.exists(video_path):
        print(f"Skipping {video_path} (file not found).")
        continue

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing {video_path} ({total} frames total)...")
    
    idx = 0
    video_saved = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened() and saved < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        if idx % FRAME_SKIP == 0:
            filename = os.path.join(OUTPUT_DIR, f"{video_name}_frame_{idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
            video_saved += 1
            if video_saved % 20 == 0:
                print(f"  Extracted {video_saved} frames from {video_name}...")
                
        idx += 1
        
    cap.release()
    print(f"Finished {video_name}. Saved {video_saved} highly diverse frames.")

print(f"\n✅ Done! Total of {saved} highly diverse, unique images saved to {OUTPUT_DIR}/")
