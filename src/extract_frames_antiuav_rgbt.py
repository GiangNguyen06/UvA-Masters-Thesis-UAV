#!/usr/bin/env python3
"""
extract_frames_antiuav_rgbt.py
-------------------------------
Pre-extract IR frames from Anti-UAV-RGBT mp4 videos to JPEG files.

This eliminates cv2.VideoCapture random-seek overhead during training.
Sequential decode of an entire video is ~10× faster than random seeks,
and subsequent JPEG reads are simple filesystem ops (~1–2 ms each).

Expected speedup during training:
  Before: ~0.6 s/batch (VideoCapture random seek over NFS/GPFS)
  After:  ~0.05–0.1 s/batch (JPEG imread from local NVMe/GPFS)

Output layout
-------------
{out_root}/{split}/{seq}/ir_{frame:06d}.jpg

Usage
-----
# Extract to the same root (adds frames/ alongside infrared.mp4):
python extract_frames_antiuav_rgbt.py \
    --src /scratch-local/$USER.$SLURM_JOB_ID/Anti-UAV-RGBT \
    --out /scratch-local/$USER.$SLURM_JOB_ID/Anti-UAV-RGBT-frames \
    --splits train val \
    --quality 95 \
    --workers 4

SLURM (add before your training command):
    python /projects/prjs2041/uav_code/extract_frames_antiuav_rgbt.py \
        --src  $LOCAL_DATA \
        --out  ${LOCAL_SCRATCH}/Anti-UAV-RGBT-frames \
        --splits train val --workers 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


# ── Worker function (runs in subprocess) ──────────────────────────────────────

def extract_sequence(args):
    """
    Extract all IR frames from one sequence video.

    Returns (seq_name, n_frames, elapsed_s, error_str|None).
    """
    seq_dir, out_seq_dir, quality = args
    seq_dir     = Path(seq_dir)
    out_seq_dir = Path(out_seq_dir)
    out_seq_dir.mkdir(parents=True, exist_ok=True)

    video_path = seq_dir / 'infrared.mp4'
    if not video_path.exists():
        return seq_dir.name, 0, 0.0, f'No infrared.mp4 in {seq_dir}'

    t0  = time.time()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return seq_dir.name, 0, 0.0, f'Cannot open {video_path}'

    n_frames = 0
    frame_idx = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_seq_dir / f'ir_{frame_idx:06d}.jpg'
        if not out_path.exists():          # skip already-extracted frames
            cv2.imwrite(str(out_path), frame, encode_params)
        n_frames += 1
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    return seq_dir.name, n_frames, elapsed, None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Pre-extract Anti-UAV-RGBT IR frames to JPEG')
    p.add_argument('--src',     required=True,
                   help='Path to Anti-UAV-RGBT root (contains train/ val/ test/)')
    p.add_argument('--out',     required=True,
                   help='Output root for extracted frames')
    p.add_argument('--splits',  nargs='+', default=['train', 'val'],
                   help='Which splits to extract (default: train val)')
    p.add_argument('--quality', type=int, default=95,
                   help='JPEG quality 1–100 (default 95 — visually lossless)')
    p.add_argument('--workers', type=int, default=4,
                   help='Parallel extraction workers (default 4)')
    return p.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src)
    out_root = Path(args.out)

    print(f'Source : {src_root}')
    print(f'Output : {out_root}')
    print(f'Splits : {args.splits}')
    print(f'Quality: {args.quality}')
    print(f'Workers: {args.workers}')
    print()

    total_frames = 0
    total_time   = 0.0
    total_errors = 0

    for split in args.splits:
        split_dir = src_root / split
        if not split_dir.exists():
            print(f'[SKIP] Split not found: {split_dir}')
            continue

        sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f'=== {split.upper()} — {len(sequences)} sequences ===')

        # Build task list for this split
        tasks = []
        for seq_dir in sequences:
            out_seq = out_root / split / seq_dir.name
            tasks.append((str(seq_dir), str(out_seq), args.quality))

        t_split_start = time.time()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract_sequence, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                seq_name, n_frames, elapsed, err = fut.result()
                total_frames += n_frames
                total_time   += elapsed
                if err:
                    total_errors += 1
                    print(f'  [{i:3d}/{len(tasks)}] ERROR {seq_name}: {err}')
                else:
                    fps = n_frames / elapsed if elapsed > 0 else 0
                    print(f'  [{i:3d}/{len(tasks)}] {seq_name:30s} '
                          f'{n_frames:5d} frames  {elapsed:5.1f}s  ({fps:.0f} fps)')

        t_split = time.time() - t_split_start
        print(f'  Split done in {t_split:.1f}s\n')

    print('=' * 60)
    print(f'Total frames extracted : {total_frames:,}')
    print(f'Total errors           : {total_errors}')
    print(f'Wall-clock time        : {total_time:.0f}s (summed, parallel actual faster)')
    print()
    print(f'Frame root to pass to training:')
    print(f'  --frames-root {out_root}')

    # Write a manifest so the dataset class can verify extraction is complete
    manifest = out_root / 'manifest.json'
    manifest_data = {
        'source':        str(src_root),
        'quality':       args.quality,
        'splits':        args.splits,
        'total_frames':  total_frames,
        'errors':        total_errors,
    }
    with open(manifest, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f'\nManifest written: {manifest}')


if __name__ == '__main__':
    main()
