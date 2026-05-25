"""
Remove near-duplicate frames from training_frames/ using perceptual hashing.
Keeps only visually unique frames (no external libraries needed - pure OpenCV + numpy).
"""

import cv2
import numpy as np
from pathlib import Path

FRAMES_DIR     = Path("backend/training_frames")
HASH_SIZE      = 16        # 16x16 perceptual hash = 256 bits
SIMILARITY_THR = 10        # Hamming distance threshold (0=identical, lower=stricter)
                           # 10 means "keep frame if it differs by >10 bits from all kept frames"


def phash(img: np.ndarray, size: int = HASH_SIZE) -> np.ndarray:
    """Compute a perceptual hash (difference hash) for an image."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    # Difference hash: compare adjacent pixels
    diff    = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.uint8)


def hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


def remove_duplicates():
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        print("[!] No frames found in", FRAMES_DIR.resolve())
        return

    print(f"[+] Scanning {len(frames)} frames for duplicates...")
    print(f"    Similarity threshold: Hamming distance > {SIMILARITY_THR} = UNIQUE")

    kept_hashes = []   # list of phash arrays for kept frames
    kept_names  = []
    removed     = []

    for i, fpath in enumerate(frames):
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"    [!] Cannot read {fpath.name} -- skipping")
            continue

        h = phash(img)

        # Check against all kept frames
        is_duplicate = False
        for kh in kept_hashes:
            if hamming(h, kh) <= SIMILARITY_THR:
                is_duplicate = True
                break

        if is_duplicate:
            fpath.unlink()   # delete the file
            removed.append(fpath.name)
        else:
            kept_hashes.append(h)
            kept_names.append(fpath.name)

        # Progress
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(frames)} -- kept {len(kept_names)}, removed {len(removed)}")

    print(f"\n{'='*50}")
    print(f"DONE!")
    print(f"  Original : {len(frames)} frames")
    print(f"  Kept     : {len(kept_names)} unique frames")
    print(f"  Removed  : {len(removed)} duplicates")
    print(f"\nUnique frames ready at: {FRAMES_DIR.resolve()}")


if __name__ == "__main__":
    remove_duplicates()
