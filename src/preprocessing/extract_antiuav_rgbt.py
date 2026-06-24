#!/usr/bin/env python3
"""
Frame Extraction Script — Anti-UAV-RGBT
Extracts frames from infrared.mp4 and visible.mp4 for all sequences
across train / val / test splits.

Output structure (mirrors the existing framecut.py convention):
  <split>/<sequence>/infrared/<sequence>_infrared_0001.jpg
  <split>/<sequence>/visible/<sequence>_visible_0001.jpg

Usage:
  /home/knguyen1/.conda/envs/uav_master/bin/python extract_antiuav_rgbt.py
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/projects/prjs2041/datasets/Anti-UAV-RGBT")
SPLITS       = ["train", "val", "test"]
MODALITIES   = ["infrared", "visible"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_video(video_path: Path, out_dir: Path, prefix: str) -> int:
    """
    Extract all frames from video_path into out_dir.
    Frames are named: <prefix>_0001.jpg, <prefix>_0002.jpg, ...
    Skips extraction if out_dir already contains the same number of frames
    as the video has (resumable).
    Returns number of frames extracted.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Resumability: skip if already fully extracted ──
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
        out_path = out_dir / f"{prefix}_{str(frame_idx).zfill(4)}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    cap.release()
    return frame_idx


def verify_alignment(seq_dir: Path) -> bool:
    """
    Check that the number of extracted IR frames equals the number of
    RGB frames (temporal alignment sanity check).
    """
    ir_frames  = len(list((seq_dir / "infrared").glob("*.jpg")))
    rgb_frames = len(list((seq_dir / "visible").glob("*.jpg")))
    return ir_frames == rgb_frames


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Frame Extraction — Anti-UAV-RGBT")
    print("=" * 50)

    total_extracted = 0
    total_skipped   = 0
    alignment_errors = []

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            print(f"\n[WARN] Split not found: {split_dir}")
            continue

        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"\n[{split.upper()}] {len(sequences)} sequences")

        for seq in tqdm(sequences, desc=f"  {split}", unit="seq"):
            for modality in MODALITIES:
                video_file = seq / f"{modality}.mp4"
                if not video_file.exists():
                    print(f"  [WARN] Missing video: {video_file}")
                    continue

                out_dir = seq / modality
                prefix  = f"{seq.name}_{modality}"

                n = extract_video(video_file, out_dir, prefix)
                if n > 0:
                    total_extracted += n
                else:
                    total_skipped += 1

            # Verify temporal alignment after both modalities extracted
            if (seq / "infrared").exists() and (seq / "visible").exists():
                if not verify_alignment(seq):
                    alignment_errors.append(str(seq))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Frames extracted : {total_extracted:,}")
    print(f"Sequences skipped (already done): {total_skipped}")

    if alignment_errors:
        print(f"\n⚠ Alignment errors ({len(alignment_errors)} sequences):")
        for e in alignment_errors:
            print(f"  - {e}")
    else:
        print("✓ All sequences: IR and RGB frame counts match")


if __name__ == "__main__":
    main()
