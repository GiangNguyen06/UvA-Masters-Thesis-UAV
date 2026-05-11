#!/usr/bin/env python3
"""
Dataset Audit Script — Anti-UAV Master's Thesis
Audits: Anti-UAV-RGBT, Anti-UAV410, ARD100, CST-AntiUAV

Reports per split:
  - Sequence count
  - Frame count
  - Annotated vs missing (exist=0) frames
  - Bounding box size distribution (tiny/small/normal/large)
  - Any issues (missing files, parse errors)
"""

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {
    "Anti-UAV-RGBT": {
        "test": "/projects/prjs2041/datasets/Anti-UAV-RGBT/test",
    },
    "Anti-UAV410": {
        "train": "/projects/prjs2041/datasets/Anti-UAV410/train",
        "val":   "/projects/prjs2041/datasets/Anti-UAV410/val",
        "test":  "/projects/prjs2041/datasets/Anti-UAV410/test",
    },
    "ARD100": {
        "all": "/projects/prjs2041/datasets/ARD100/annotations",
    },
    "CST-AntiUAV": {
        "train": "/projects/prjs2041/datasets/CST-AntiUAV-full/CST-AntiUAV/CST-AntiUAV/train",
        "val":   "/projects/prjs2041/datasets/CST-AntiUAV-full/CST-AntiUAV/CST-AntiUAV/val",
        "test":  "/projects/prjs2041/datasets/CST-AntiUAV-full/CST-AntiUAV/CST-AntiUAV/test",
    },
}

# ── Size classification (by bounding box diagonal, in pixels) ─────────────────
def diagonal(w, h):
    return math.sqrt(w ** 2 + h ** 2)

def classify(d):
    if d <= 10:  return "tiny"
    if d <= 30:  return "small"
    if d <= 50:  return "normal"
    return "large"

def empty_stats():
    return {
        "sequences": 0,
        "frames":    0,
        "annotated": 0,   # exist=1
        "missing":   0,   # exist=0
        "size_dist": defaultdict(int),
        "issues":    [],
    }

# ── Anti-UAV-RGBT (paired RGB + IR per-sequence JSON) ────────────────────────
def audit_rgbt_dataset(split_paths):
    """
    Reads both infrared.json and visible.json per sequence.
    Reports IR and RGB stats separately, and flags frame count mismatches
    between the two modalities (critical for Teacher-Student alignment).
    """
    results = {}
    for split, path in split_paths.items():
        s = empty_stats()
        s["rgb_frames"]    = 0
        s["rgb_annotated"] = 0
        s["rgb_size_dist"] = defaultdict(int)
        s["modality_mismatches"] = []  # sequences where IR/RGB frame counts differ

        path = Path(path)
        if not path.exists():
            s["issues"].append(f"Root path not found: {path}")
            results[split] = s
            continue

        sequences = sorted(d for d in path.iterdir() if d.is_dir())
        s["sequences"] = len(sequences)

        for seq in sequences:
            ir_file  = seq / "infrared.json"
            rgb_file = seq / "visible.json"

            # ── IR stream ──
            if not ir_file.exists():
                s["issues"].append(f"Missing infrared.json: {seq.name}")
                ir_exist = []
            else:
                try:
                    with open(ir_file) as f:
                        ir_data = json.load(f)
                    ir_exist   = ir_data.get("exist",   [])
                    ir_gt_rect = ir_data.get("gt_rect", [])
                    s["frames"]    += len(ir_exist)
                    s["annotated"] += sum(ir_exist)
                    s["missing"]   += len(ir_exist) - sum(ir_exist)
                    for e, box in zip(ir_exist, ir_gt_rect):
                        if e == 1 and box and len(box) == 4:
                            d = diagonal(box[2], box[3])
                            s["size_dist"][classify(d)] += 1
                except Exception as e:
                    s["issues"].append(f"IR JSON error in {seq.name}: {e}")
                    ir_exist = []

            # ── RGB stream ──
            if not rgb_file.exists():
                s["issues"].append(f"Missing visible.json: {seq.name}")
            else:
                try:
                    with open(rgb_file) as f:
                        rgb_data = json.load(f)
                    rgb_exist   = rgb_data.get("exist",   [])
                    rgb_gt_rect = rgb_data.get("gt_rect", [])
                    s["rgb_frames"]    += len(rgb_exist)
                    s["rgb_annotated"] += sum(rgb_exist)
                    for e, box in zip(rgb_exist, rgb_gt_rect):
                        if e == 1 and box and len(box) == 4:
                            d = diagonal(box[2], box[3])
                            s["rgb_size_dist"][classify(d)] += 1

                    # ── Check temporal alignment ──
                    if len(ir_exist) != len(rgb_exist):
                        s["modality_mismatches"].append(
                            f"{seq.name}: IR={len(ir_exist)} frames, RGB={len(rgb_exist)} frames"
                        )
                except Exception as e:
                    s["issues"].append(f"RGB JSON error in {seq.name}: {e}")

        results[split] = s
    return results

# ── Anti-UAV410 (IR only, per-sequence JSON) ──────────────────────────────────
def audit_json_dataset(split_paths, ann_filename):
    results = {}
    for split, path in split_paths.items():
        s = empty_stats()
        path = Path(path)
        if not path.exists():
            s["issues"].append(f"Root path not found: {path}")
            results[split] = s
            continue

        sequences = sorted(d for d in path.iterdir() if d.is_dir())
        s["sequences"] = len(sequences)

        for seq in sequences:
            ann_file = seq / ann_filename
            if not ann_file.exists():
                s["issues"].append(f"Missing annotation in: {seq.name}")
                continue

            try:
                with open(ann_file) as f:
                    data = json.load(f)
            except Exception as e:
                s["issues"].append(f"JSON parse error in {seq.name}: {e}")
                continue

            exist   = data.get("exist",   [])
            gt_rect = data.get("gt_rect", [])

            s["frames"]    += len(exist)
            s["annotated"] += sum(exist)
            s["missing"]   += len(exist) - sum(exist)

            for e, box in zip(exist, gt_rect):
                if e == 1 and box and len(box) == 4:
                    d = diagonal(box[2], box[3])
                    s["size_dist"][classify(d)] += 1

        results[split] = s
    return results

# ── ARD100 (Pascal VOC XML, one file per frame) ───────────────────────────────
def audit_ard100(ann_root):
    s = empty_stats()
    path = Path(ann_root)
    if not path.exists():
        s["issues"].append(f"Root path not found: {path}")
        return {"all": s}

    sequences = sorted(d for d in path.iterdir() if d.is_dir())
    s["sequences"] = len(sequences)

    for seq in sequences:
        xml_files = sorted(seq.glob("*.xml"))
        s["frames"] += len(xml_files)

        for xf in xml_files:
            try:
                root = ET.parse(xf).getroot()
                obj  = root.find("object")
                if obj is None:
                    s["missing"] += 1
                    continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                d = diagonal(xmax - xmin, ymax - ymin)
                s["size_dist"][classify(d)] += 1
                s["annotated"] += 1
            except Exception as e:
                s["issues"].append(f"{xf.name}: {e}")
                s["missing"] += 1

    return {"all": s}

# ── CST-AntiUAV (gt.txt + IR_label.json) ─────────────────────────────────────
def audit_cst(split_paths):
    results = {}
    for split, path in split_paths.items():
        s = empty_stats()
        path = Path(path)
        if not path.exists():
            s["issues"].append(f"Root path not found: {path}")
            results[split] = s
            continue

        sequences = sorted(d for d in path.iterdir() if d.is_dir())
        s["sequences"] = len(sequences)

        for seq in sequences:
            # ── exist flags: prefer IR_label.json, fall back to exist.txt ──
            ir_json    = seq / "IR_label.json"
            exist_txt  = seq / "exist.txt"
            gt_txt     = seq / "gt.txt"

            if ir_json.exists():
                try:
                    with open(ir_json) as f:
                        exist = json.load(f).get("exist", [])
                except Exception as e:
                    s["issues"].append(f"JSON error in {seq.name}: {e}")
                    continue
            elif exist_txt.exists():
                with open(exist_txt) as f:
                    exist = [int(l.strip()) for l in f if l.strip()]
            else:
                s["issues"].append(f"No exist annotation: {seq.name}")
                continue

            s["frames"]    += len(exist)
            s["annotated"] += sum(exist)
            s["missing"]   += len(exist) - sum(exist)

            # ── bounding boxes ──
            if gt_txt.exists():
                with open(gt_txt) as f:
                    for line, e in zip(f, exist):
                        if e != 1:
                            continue
                        parts = line.strip().split(",")
                        if len(parts) == 4:
                            try:
                                w, h = float(parts[2]), float(parts[3])
                                d = diagonal(w, h)
                                s["size_dist"][classify(d)] += 1
                            except ValueError:
                                pass
            else:
                s["issues"].append(f"Missing gt.txt: {seq.name}")

        results[split] = s
    return results

# ── Pretty printer ────────────────────────────────────────────────────────────
CATS = ["tiny", "small", "normal", "large"]

def print_size_dist(size_dist, indent="  │    "):
    size_total = sum(size_dist.values())
    for cat in CATS:
        n   = size_dist.get(cat, 0)
        pct = 100 * n / size_total if size_total else 0
        bar = "█" * int(pct / 2)
        print(f"{indent}{cat:7s}: {n:7,}  ({pct:5.1f}%)  {bar}")

def print_results(dataset_name, results, is_rgbt=False):
    print(f"\n{'═'*62}")
    print(f"  {dataset_name}")
    print(f"{'═'*62}")
    for split, s in results.items():
        total_fr  = s["frames"]
        total_ann = s["annotated"]
        vis_pct   = 100 * total_ann / total_fr if total_fr else 0
        print(f"\n  ┌─ [{split}]")
        print(f"  │  Sequences : {s['sequences']}")

        if is_rgbt:
            # IR stream
            print(f"  │  ── IR stream (Student input)")
            print(f"  │    Frames   : {total_fr:,}")
            print(f"  │    Visible  : {total_ann:,}  ({vis_pct:.1f}%)")
            print(f"  │    Occluded : {s['missing']:,}  ({100-vis_pct:.1f}%)")
            print(f"  │    Size distribution:")
            print_size_dist(s["size_dist"])

            # RGB stream
            rgb_fr  = s.get("rgb_frames", 0)
            rgb_ann = s.get("rgb_annotated", 0)
            rgb_vis_pct = 100 * rgb_ann / rgb_fr if rgb_fr else 0
            print(f"  │  ── RGB stream (Teacher input)")
            print(f"  │    Frames   : {rgb_fr:,}")
            print(f"  │    Visible  : {rgb_ann:,}  ({rgb_vis_pct:.1f}%)")
            print(f"  │    Size distribution:")
            print_size_dist(s.get("rgb_size_dist", {}))

            # Alignment check
            mismatches = s.get("modality_mismatches", [])
            if mismatches:
                print(f"  │  ⚠ Temporal mismatches ({len(mismatches)} sequences):")
                for m in mismatches[:5]:
                    print(f"  │    - {m}")
            else:
                print(f"  │  ✓ All sequences temporally aligned (IR/RGB frame counts match)")
        else:
            print(f"  │  Frames    : {total_fr:,}")
            print(f"  │  Visible   : {total_ann:,}  ({vis_pct:.1f}%)")
            print(f"  │  Occluded  : {s['missing']:,}  ({100-vis_pct:.1f}%)")
            print(f"  │  Size distribution (by bbox diagonal):")
            print_size_dist(s["size_dist"])

        if s["issues"]:
            print(f"  │  ⚠ Issues ({len(s['issues'])}):")
            for issue in s["issues"][:5]:
                print(f"  │    - {issue}")
            if len(s["issues"]) > 5:
                print(f"  │    ... and {len(s['issues'])-5} more")
        print(f"  └─")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Dataset Audit — Anti-UAV Master's Thesis")
    print(f"{'═'*62}")

    print_results(
        "Anti-UAV-RGBT  (T1 Source — RGB+IR video)",
        audit_rgbt_dataset(PATHS["Anti-UAV-RGBT"]),
        is_rgbt=True,
    )

    print_results(
        "Anti-UAV410  (T1 Target — IR frames)",
        audit_json_dataset(PATHS["Anti-UAV410"], ann_filename="IR_label.json"),
    )

    print_results(
        "ARD100  (Ego-Motion Stress Test — RGB video)",
        audit_ard100(PATHS["ARD100"]["all"]),
    )

    print_results(
        "CST-AntiUAV  (T2 Extreme Scale — IR frames)",
        audit_cst(PATHS["CST-AntiUAV"]),
    )

    print(f"\n{'═'*62}")
    print("Audit complete.")

if __name__ == "__main__":
    main()
