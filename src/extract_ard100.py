#!/usr/bin/env python3
"""
Frame Extraction Script — ARD100
Extracts frames from train_videos/ and test_videos/ MP4 files.

Output structure (consistent with annotation filenames):
  images/train/phantom02/phantom02_0001.jpg
  images/test/phantom02/phantom02_0001.jpg

The frame numbering matches the XML annotation filenames exactly:
  annotations/phantom02/phantom02_0001.xml  ↔  images/train/phantom02/phantom02_0001.jpg

Usage:
  /home/knguyen1/.conda/envs/uav_master/bin/python extract_ard100.py
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/projects/prjs2041/datasets/ARD100")
SPLITS = {
    "train": DATASET_ROOT / "train_videos",
    "test":  DATASET_ROOT / "test_videos",
}
OUTPUT_ROOT = DATASET_ROOT / "images"

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_video(video_path: Path, out_dir: Path) -> int:
    """
    Extract all frames from video_path into out_dir.
    Naming: <video_stem>_0001.jpg, <video_stem>_0002.jpg, ...
    This matches the annotation XML naming convention exactly.
    Resumable: skips if frame count already matches video length.
    Returns number of frames extracted (0 if skipped).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = video_path.stem  # e.g. "phantom02"

    # ── Resumability ──
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) == total_frames:
        cap.release()
        return 0  # already done

    # ── Extract ──
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        out_path = out_dir / f"{stem}_{str(frame_idx).zfill(4)}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    cap.release()
    return frame_idx


def verify_against_annotations(split: str, seq_name: str, out_dir: Path) -> bool:
    """
    Check that the number of extracted frames matches the number of
    XML annotation files for this sequence.
    """
    ann_dir = DATASET_ROOT / "annotations" / seq_name
    if not ann_dir.exists():
        return True  # no annotations to check against (test set may differ)

    n_xml    = len(list(ann_dir.glob("*.xml")))
    n_frames = len(list(out_dir.glob("*.jpg")))
    return n_xml == n_frames


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Frame Extraction — ARD100")
    print("=" * 50)

    total_extracted  = 0
    total_skipped    = 0
    annotation_mismatches = []

    for split, video_dir in SPLITS.items():
        if not video_dir.exists():
            print(f"\n[WARN] Video folder not found: {video_dir}")
            continue

        videos = sorted(video_dir.glob("*.mp4"))
        print(f"\n[{split.upper()}] {len(videos)} videos")

        for video_path in tqdm(videos, desc=f"  {split}", unit="video"):
            seq_name = video_path.stem  # e.g. "phantom02"
            out_dir  = OUTPUT_ROOT / split / seq_name

            n = extract_video(video_path, out_dir)
            if n > 0:
                total_extracted += n
            else:
                total_skipped += 1

            # Verify frame count matches annotation count
            if not verify_against_annotations(split, seq_name, out_dir):
                annotation_mismatches.append(
                    f"{split}/{seq_name}: "
                    f"{len(list(out_dir.glob('*.jpg')))} frames vs "
                    f"{len(list((DATASET_ROOT / 'annotations' / seq_name).glob('*.xml')))} XMLs"
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Frames extracted : {total_extracted:,}")
    print(f"Videos skipped (already done): {total_skipped}")

    if annotation_mismatches:
        print(f"\n⚠ Frame/annotation count mismatches ({len(annotation_mismatches)}):")
        for m in annotation_mismatches:
            print(f"  - {m}")
    else:
        print("✓ All sequences: frame counts match annotation counts")


if __name__ == "__main__":
    main()
