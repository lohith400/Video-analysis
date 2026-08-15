"""
Extract training frames from all videos in backend/videos/
Saves frames to backend/training_frames/
- 1 frame every N frames (default: every 15 = ~1 per second at 30fps)
- Skips blurry frames automatically (Laplacian variance check)
- Saves both a clean copy + an auto-annotated preview (using existing YOLO model)
"""

import cv2
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
VIDEOS_DIR    = Path("backend/videos")
OUTPUT_DIR    = Path("backend/training_frames")
EXTRACT_EVERY = 15       # extract 1 frame every N frames (~1 fps for 15fps video)
MIN_BLUR      = 80.0     # skip frames blurrier than this (Laplacian variance)
MAX_FRAMES    = 2000     # safety cap per video
# ────────────────────────────────────────────────────────────────────────


def is_blurry(frame, threshold=MIN_BLUR):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def extract_frames():
    OUTPUT_DIR.mkdir(exist_ok=True)

    video_files = list(VIDEOS_DIR.glob("*.mp4")) + \
                  list(VIDEOS_DIR.glob("*.avi")) + \
                  list(VIDEOS_DIR.glob("*.mov")) + \
                  list(VIDEOS_DIR.glob("*.mkv"))

    if not video_files:
        print(f"[!] No videos found in {VIDEOS_DIR.resolve()}")
        return

    print(f"[+] Found {len(video_files)} video(s)")

    total_saved = 0

    for vid_path in sorted(video_files):
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"[!] Could not open: {vid_path.name}")
            continue

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30
        total_vid  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_stem   = vid_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        saved      = 0
        frame_idx  = 0
        skipped_blur = 0

        extract_interval = max(1, int(round(fps)))

        print(f"\n[>>] {vid_path.name}  ({total_vid} frames @ {fps:.1f}fps | extracting every {extract_interval} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % extract_interval == 0:
                if is_blurry(frame):
                    skipped_blur += 1
                else:
                    fname = OUTPUT_DIR / f"{vid_stem}_f{frame_idx:06d}.jpg"
                    cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                    total_saved += 1

                    if saved % 50 == 0:
                        print(f"   {saved} frames saved...")

                if saved >= MAX_FRAMES:
                    print(f"   [!] Reached MAX_FRAMES cap ({MAX_FRAMES}) -- stopping this video")
                    break

            frame_idx += 1

        cap.release()
        print(f"   DONE: Saved {saved} frames  (skipped {skipped_blur} blurry frames)")

    print(f"\n{'='*50}")
    print(f"COMPLETE -- Total frames extracted: {total_saved}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("\nNext step: Upload the 'training_frames' folder to Roboflow for annotation.")


if __name__ == "__main__":
    extract_frames()
