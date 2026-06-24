#!/usr/bin/env python3
"""
JSON → YOLO Label Conversion
Converts Anti-UAV-RGBT and Anti-UAV410 JSON annotations to YOLO .txt format.

Input format (per sequence):
  infrared.json / IR_label.json:
    {
      "exist":   [1, 1, 0, 1, ...],
      "gt_rect": [[x, y, w, h], [x, y, w, h], ...]
    }

Output format (one .txt per frame, skipped if exist=0):
  labels/<sequence>/<sequence>_0001.txt
  Content: "0 x_center y_center width height"  (all normalized 0-1)

Usage:
  /home/knguyen1/.conda/envs/uav_master/bin/python json2yolo.py
"""

import json
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASETS = {
    "Anti-UAV-RGBT": {
        "root":       Path("/projects/prjs2041/datasets/Anti-UAV-RGBT"),
        "splits":     ["train", "val", "test"],
        "ann_file":   "infrared.json",   # IR stream (Student input)
        "img_w":      640,               # IR sensor resolution
        "img_h":      512,
    },
    "Anti-UAV410": {
        "root":       Path("/projects/prjs2041/datasets/Anti-UAV410"),
        "splits":     ["train", "val", "test"],
        "ann_file":   "IR_label.json",
        "img_w":      640,
        "img_h":      512,
    },
}

# ── Conversion ────────────────────────────────────────────────────────────────
def xywh_to_yolo(x, y, w, h, img_w, img_h):
    """
    Convert pixel [x_topleft, y_topleft, w, h] to
    normalised YOLO [x_center, y_center, w, h].
    """
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w_norm   = w / img_w
    h_norm   = h / img_h
    # Clamp to [0, 1] to handle any annotation boundary issues
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm   = max(0.0, min(1.0, w_norm))
    h_norm   = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


def convert_sequence(seq_dir: Path, ann_file: str, labels_dir: Path,
                     img_w: int, img_h: int):
    """Convert one sequence's JSON annotations to per-frame YOLO .txt files."""
    ann_path = seq_dir / ann_file
    if not ann_path.exists():
        return 0, 0  # (frames_written, frames_skipped)

    with open(ann_path) as f:
        data = json.load(f)

    exist   = data.get("exist",   [])
    gt_rect = data.get("gt_rect", [])

    if len(exist) != len(gt_rect):
        print(f"  [WARN] Length mismatch in {seq_dir.name}: "
              f"exist={len(exist)}, gt_rect={len(gt_rect)}")
        return 0, 0

    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for frame_idx, (e, box) in enumerate(zip(exist, gt_rect)):
        frame_num  = str(frame_idx + 1).zfill(4)
        label_path = labels_dir / f"{seq_dir.name}_{frame_num}.txt"

        if e == 0 or box is None or len(box) != 4:
            # Drone not visible — write empty label file (YOLO expects this)
            label_path.write_text("")
            skipped += 1
            continue

        x, y, w, h = box

        # Skip degenerate boxes
        if w <= 0 or h <= 0:
            label_path.write_text("")
            skipped += 1
            continue

        xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w, img_h)
        label_path.write_text(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
        written += 1

    return written, skipped


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("JSON → YOLO Label Conversion")
    print("=" * 50)

    for dataset_name, cfg in DATASETS.items():
        root     = cfg["root"]
        splits   = cfg["splits"]
        ann_file = cfg["ann_file"]
        img_w    = cfg["img_w"]
        img_h    = cfg["img_h"]

        print(f"\n── {dataset_name} ──")

        total_written = 0
        total_skipped = 0

        for split in splits:
            split_dir = root / split
            if not split_dir.exists():
                print(f"  [WARN] Split not found: {split_dir}")
                continue

            sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())
            labels_root = root / "labels" / split

            split_written = 0
            split_skipped = 0

            for seq in tqdm(sequences, desc=f"  {split:5s}", unit="seq"):
                labels_dir = labels_root / seq.name
                w, s = convert_sequence(seq, ann_file, labels_dir, img_w, img_h)
                split_written += w
                split_skipped += s

            print(f"  [{split}] {split_written:,} annotated frames, "
                  f"{split_skipped:,} empty (exist=0)")
            total_written += split_written
            total_skipped += split_skipped

        print(f"  Total: {total_written:,} labels written, "
              f"{total_skipped:,} empty frames")

    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
