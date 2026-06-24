#!/usr/bin/env python3
"""
ARD100 Data Preparation Script
Combines three steps into one:
  1. Create ImageSets/Main/ train.txt and test.txt from the known split
  2. Convert VOC XML annotations to YOLO .txt format
  3. Generate image path lists (train_images.txt, test_images.txt)

The train/test split is the official one from the YOLOMG paper (generate_mask5.py).

Usage:
  /home/knguyen1/.conda/envs/uav_master/bin/python prepare_ard100.py
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/projects/prjs2041/datasets/ARD100")
ANN_DIR      = DATASET_ROOT / "annotations"
LABELS_DIR   = DATASET_ROOT / "labels"
IMAGES_DIR   = DATASET_ROOT / "images"
IMAGESETS_DIR = DATASET_ROOT / "ImageSets" / "Main"

CLASSES = ["Drone"]  # ARD100 only has one class

# ── Official train/test split from YOLOMG paper ───────────────────────────────
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

# ── VOC → YOLO conversion ─────────────────────────────────────────────────────
def convert_box(size, box):
    """Convert VOC [xmin, xmax, ymin, ymax] to YOLO [x_c, y_c, w, h] normalised."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_xml_to_yolo(xml_path: Path, label_path: Path):
    """Convert a single VOC XML file to YOLO .txt format."""
    label_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  [WARN] Parse error {xml_path.name}: {e}")
        return

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    lines = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue
        cls = obj.find('name').text
        # Handle both "Drone" and "drone" and "UAV" labels
        if cls.lower() not in [c.lower() for c in CLASSES]:
            continue
        cls_id = 0  # single class

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        xmax = float(bbox.find('xmax').text)
        ymin = float(bbox.find('ymin').text)
        ymax = float(bbox.find('ymax').text)

        # Clamp to image boundaries
        xmin = max(1.0, xmin)
        ymin = max(1.0, ymin)
        xmax = min(float(w), xmax)
        ymax = min(float(h), ymax)

        bb = convert_box((w, h), (xmin, xmax, ymin, ymax))
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in bb))

    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))


# ── Step 1: Create ImageSets ──────────────────────────────────────────────────
def create_imagesets():
    print("\n[Step 1] Creating ImageSets/Main/ from official YOLOMG split...")
    IMAGESETS_DIR.mkdir(parents=True, exist_ok=True)

    for split, seqs in [("train", TRAIN_SEQS), ("test", TEST_SEQS)]:
        frame_ids = []
        for seq in seqs:
            seq_ann_dir = ANN_DIR / seq
            if not seq_ann_dir.exists():
                print(f"  [WARN] Annotation folder not found: {seq_ann_dir}")
                continue
            for xml_file in sorted(seq_ann_dir.glob("*.xml")):
                # ID format: phantom02/phantom02_0001
                frame_ids.append(f"{seq}/{xml_file.stem}")

        out_path = IMAGESETS_DIR / f"{split}.txt"
        with open(out_path, 'w') as f:
            f.write('\n'.join(frame_ids))
        print(f"  {split}.txt → {len(frame_ids):,} frames")


# ── Step 2: Convert VOC XML → YOLO .txt ──────────────────────────────────────
def convert_all_annotations():
    print("\n[Step 2] Converting VOC XML → YOLO .txt labels...")

    for split, seqs in [("train", TRAIN_SEQS), ("test", TEST_SEQS)]:
        print(f"  [{split}]")
        for seq in tqdm(seqs, desc=f"    {split}", unit="seq"):
            seq_ann_dir = ANN_DIR / seq
            if not seq_ann_dir.exists():
                continue
            for xml_file in sorted(seq_ann_dir.glob("*.xml")):
                label_path = LABELS_DIR / seq / (xml_file.stem + ".txt")
                convert_xml_to_yolo(xml_file, label_path)


# ── Step 3: Generate image path lists ────────────────────────────────────────
def generate_image_lists():
    print("\n[Step 3] Generating image path lists...")

    for split, seqs in [("train", TRAIN_SEQS), ("test", TEST_SEQS)]:
        image_paths = []
        for seq in seqs:
            split_dir = IMAGES_DIR / split / seq
            if not split_dir.exists():
                print(f"  [WARN] Images not yet extracted for {seq} — run extract_ard100.py first")
                continue
            for jpg in sorted(split_dir.glob("*.jpg")):
                image_paths.append(str(jpg))

        out_path = DATASET_ROOT / f"{split}_images.txt"
        with open(out_path, 'w') as f:
            f.write('\n'.join(image_paths))
        print(f"  {split}_images.txt → {len(image_paths):,} image paths")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("ARD100 Data Preparation")
    print("=" * 50)
    create_imagesets()
    convert_all_annotations()
    generate_image_lists()
    print("\n" + "=" * 50)
    print("Done. Next step: run generate_mask5_fixed.py to create motion masks.")


if __name__ == "__main__":
    main()
