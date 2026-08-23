from ultralytics import YOLO
import os

path = "models/helmet_detector.pt"

if not os.path.isfile(path):
    print("❌ File not found:", path)
else:
    m = YOLO(path)
    print("✔ Model loaded")
    print("Classes:", m.names)
    # Should show something like:
    # {0: 'helmet', 1: 'no_helmet'}  ← good
    # or {0: 'with_helmet', 1: 'without_helmet'}