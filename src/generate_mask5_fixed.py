#!/usr/bin/env python3
"""
Motion Mask Generation for ARD100 — Fixed for Snellius
Based on generate_mask5.py from the YOLOMG codebase.

Generates pixel-level motion difference maps (mask32) for both
train and test video sequences. These masks are YOLOMG's second
input alongside the RGB frames.

Requires FD5_mask.py to be present in the same directory.

Usage:
  /home/knguyen1/.conda/envs/uav_master/bin/python generate_mask5_fixed.py
"""

import cv2
from pathlib import Path
from tqdm import tqdm
from FD5_mask import FD5_mask

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/projects/prjs2041/datasets/ARD100")
TRAIN_VIDEOS = DATASET_ROOT / "train_videos"
TEST_VIDEOS  = DATASET_ROOT / "test_videos"

# ── Official train/test split ─────────────────────────────────────────────────
TRAIN_SEQS = [
    'phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20',
    'phantom28', 'phantom29', 'phantom30', 'phantom32', 'phantom36', 'phantom40',
    'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65',
    'phantom66', 'phantom68', 'phantom70', 'phantom71', 'phantom74', 'phantom75',
    'phantom76', 'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
    'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90',
    'phantom101', 'phantom103', 'phantom104', 'phantom105', 'phantom106', 'phantom107',
    'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115',
    'phantom116', 'phantom117', 'phantom118', 'phantom120', 'phantom132', 'phantom137',
    'phantom138', 'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
    'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150',
]

TEST_SEQS = [
    'phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22',
    'phantom39', 'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54',
    'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60', 'phantom61',
    'phantom64', 'phantom73', 'phantom79', 'phantom92', 'phantom93', 'phantom94',
    'phantom95', 'phantom97', 'phantom102', 'phantom110', 'phantom113', 'phantom119',
    'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144',
]

# ── Mask generation ───────────────────────────────────────────────────────────
def process_sequence(video_path: Path, video_name: str):
    """Generate motion masks for a single video sequence."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open: {video_path}")
        return

    lastFrame1 = None
    lastFrame2 = None
    lastFrame3 = None
    lastFrame4 = None
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        if lastFrame4 is None:
            if lastFrame1 is None:
                lastFrame1 = frame
            elif lastFrame2 is None:
                lastFrame2 = frame
            elif lastFrame3 is None:
                lastFrame3 = frame
            else:
                lastFrame4 = frame
            continue

        FD5_mask(lastFrame1, lastFrame3, frame, video_name, count - 2)

        lastFrame1 = lastFrame2
        lastFrame2 = lastFrame3
        lastFrame3 = lastFrame4
        lastFrame4 = frame

    cap.release()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Motion Mask Generation — ARD100")
    print("=" * 50)

    print(f"\n[TRAIN] {len(TRAIN_SEQS)} sequences")
    for seq in tqdm(TRAIN_SEQS, desc="  train", unit="seq"):
        video_path = TRAIN_VIDEOS / f"{seq}.mp4"
        if not video_path.exists():
            print(f"  [WARN] Video not found: {video_path}")
            continue
        process_sequence(video_path, seq)

    print(f"\n[TEST] {len(TEST_SEQS)} sequences")
    for seq in tqdm(TEST_SEQS, desc="  test", unit="seq"):
        video_path = TEST_VIDEOS / f"{seq}.mp4"
        if not video_path.exists():
            print(f"  [WARN] Video not found: {video_path}")
            continue
        process_sequence(video_path, seq)

    print("\n" + "=" * 50)
    print("Mask generation complete.")


if __name__ == "__main__":
    main()
