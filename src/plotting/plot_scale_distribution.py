#!/usr/bin/env python3
"""
plot_scale_distribution.py
--------------------------
Computes and plots the UAV target-size distribution across the three
training datasets used in the sequential continual learning curriculum:

    Stage 1 — Anti-UAV-RGBT   (standard-scale targets, IR video)
    Stage 2 — Anti-UAV410     (thermal domain, IR frames)
    Stage 3 — CST Anti-UAV    (extreme tiny targets, IR video)

Size bins match eval_full_analysis.py (corrected UAV-specific thresholds,
based on the longest bounding-box side in pixels):

    Tiny   :  max(w, h) <  16 px
    Small  :  max(w, h) <  32 px
    Normal :  max(w, h) <  64 px
    Large  :  max(w, h) >= 64 px

Output
------
  - Console: per-dataset counts and percentages
  - File:    scale_distribution.png  (grouped bar chart, thesis-ready)

Usage (on Snellius, single A100, ~0.5h, ~16 SBU)
------
    python plot_scale_distribution.py \
        --rgbt-root  /projects/prjs2041/datasets/Anti-UAV-RGBT \
        --uav410-root /projects/prjs2041/datasets/Anti-UAV410 \
        --cst-root   /projects/prjs2041/datasets/CST-AntiUAV-full/CST-AntiUAV/CST-AntiUAV \
        --splits     train val \
        --out        /projects/prjs2041/analysis/scale_distribution.png
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed on Snellius
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Size bins ─────────────────────────────────────────────────────────────────
BINS   = ["Tiny", "Small", "Normal", "Large"]
COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]   # red → blue

def classify(w, h):
    """Bin by bounding-box AREA in px^2, matching the per-stratum mAP and the
    SSH buffer (thresholds 256/1024/4096 = 16^2/32^2/64^2). All datasets are
    native 640x512, so raw box w*h is the same frame used elsewhere."""
    area = w * h
    if area < 256:   return "Tiny"
    if area < 1024:  return "Small"
    if area < 4096:  return "Normal"
    return "Large"


# ── Parsers ───────────────────────────────────────────────────────────────────

def _count_from_json(ann_file, gt_key="gt_rect"):
    """
    Parse a JSON annotation file that has 'exist' and a gt key.
    Returns a Counter-like dict of {bin: count}.
    """
    counts = defaultdict(int)
    try:
        with open(ann_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot parse {ann_file}: {e}", file=sys.stderr)
        return counts

    exist   = data.get("exist", [])
    gt_rect = data.get(gt_key, data.get("gt_rect", []))  # CST uses 'gt'

    for e, box in zip(exist, gt_rect):
        if e != 1 or not box or len(box) < 4:
            continue
        w, h = float(box[2]), float(box[3])
        if w > 0 and h > 0:
            counts[classify(w, h)] += 1
    return counts


def _count_from_gt_txt(gt_txt):
    """Parse CST gt.txt fallback (x y w h, comma or space separated)."""
    counts = defaultdict(int)
    try:
        with open(gt_txt) as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if len(parts) < 4:
                    continue
                w, h = float(parts[2]), float(parts[3])
                if w > 0 and h > 0:
                    counts[classify(w, h)] += 1
    except Exception as e:
        print(f"  [WARN] Cannot parse {gt_txt}: {e}", file=sys.stderr)
    return counts


def collect_rgbt(root, splits):
    """Anti-UAV-RGBT: {root}/{split}/{seq}/infrared.json  gt_rect=[x,y,w,h]"""
    counts = defaultdict(int)
    root = Path(root)
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [WARN] RGBT split not found: {split_dir}", file=sys.stderr)
            continue
        for seq in sorted(split_dir.iterdir()):
            if not seq.is_dir():
                continue
            ann = seq / "infrared.json"
            if ann.exists():
                for k, v in _count_from_json(ann, gt_key="gt_rect").items():
                    counts[k] += v
    return counts


def collect_uav410(root, splits):
    """Anti-UAV410: {root}/{split}/{seq}/IR_label.json  gt_rect=[x,y,w,h]"""
    counts = defaultdict(int)
    root = Path(root)
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [WARN] UAV410 split not found: {split_dir}", file=sys.stderr)
            continue
        for seq in sorted(split_dir.iterdir()):
            if not seq.is_dir():
                continue
            ann = seq / "IR_label.json"
            if ann.exists():
                for k, v in _count_from_json(ann, gt_key="gt_rect").items():
                    counts[k] += v
    return counts


def collect_cst(root, splits):
    """
    CST Anti-UAV: {root}/{split}/{seq}/IR_label.json  (key='gt', not 'gt_rect')
    Falls back to gt.txt if IR_label.json absent.
    """
    counts = defaultdict(int)
    root = Path(root)
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [WARN] CST split not found: {split_dir}", file=sys.stderr)
            continue
        for seq in sorted(split_dir.iterdir()):
            if not seq.is_dir():
                continue
            ir_json = seq / "IR_label.json"
            gt_txt  = seq / "gt.txt"
            if ir_json.exists():
                for k, v in _count_from_json(ir_json, gt_key="gt").items():
                    counts[k] += v
            elif gt_txt.exists():
                for k, v in _count_from_gt_txt(gt_txt).items():
                    counts[k] += v
    return counts


# ── Plotting ──────────────────────────────────────────────────────────────────

def print_table(name, counts):
    total = sum(counts.values())
    print(f"\n{name}  (total annotated frames: {total:,})")
    print(f"  {'Bin':<8}  {'Count':>8}  {'%':>6}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*6}")
    for b in BINS:
        n = counts.get(b, 0)
        pct = 100 * n / total if total else 0
        print(f"  {b:<8}  {n:>8,}  {pct:>5.1f}%")


def make_figure(datasets, out_path):
    """
    datasets: list of (label, counts_dict)
    Produces a grouped bar chart — one group per dataset, one bar per size bin.
    """
    labels = [d[0] for d in datasets]
    n_ds   = len(labels)
    n_bins = len(BINS)

    # Convert to percentage arrays
    pcts = []
    for _, counts in datasets:
        total = sum(counts.values()) or 1
        pcts.append([100 * counts.get(b, 0) / total for b in BINS])

    x     = np.arange(n_ds)
    width = 0.18
    offsets = np.linspace(-(n_bins - 1) / 2, (n_bins - 1) / 2, n_bins) * width

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (bin_name, color, offset) in enumerate(zip(BINS, COLORS, offsets)):
        values = [pcts[j][i] for j in range(n_ds)]
        bars = ax.bar(x + offset, values, width, label=bin_name,
                      color=color, edgecolor="white", linewidth=0.5)
        # Label bars if pct >= 3%
        for bar, val in zip(bars, values):
            if val >= 3.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=7.5, color="#333333"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Proportion of annotated frames (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "UAV Target-Size Distribution Across Training Datasets\n"
        r"(bin = bbox area: Tiny $<$256, Small 256–1024, Normal 1024–4096, Large $\geq$4096 px$^2$)",
        fontsize=10
    )
    ax.legend(title="Size bin", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgbt-root",  required=True,
                   help="Root of Anti-UAV-RGBT dataset")
    p.add_argument("--uav410-root", required=True,
                   help="Root of Anti-UAV410 dataset")
    p.add_argument("--cst-root",   required=True,
                   help="Root of CST Anti-UAV dataset")
    p.add_argument("--splits",     nargs="+", default=["train", "val"],
                   help="Which splits to include (default: train val)")
    p.add_argument("--out",        default="scale_distribution.png",
                   help="Output figure path")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("UAV Scale Distribution")
    print(f"Splits: {args.splits}")
    print("=" * 60)

    print("\n[1/3] Anti-UAV-RGBT (Stage 1) ...")
    rgbt  = collect_rgbt(args.rgbt_root, args.splits)
    print_table("Anti-UAV-RGBT  (Stage 1 — standard-scale targets)", rgbt)

    print("\n[2/3] Anti-UAV410 (Stage 2) ...")
    uav410 = collect_uav410(args.uav410_root, args.splits)
    print_table("Anti-UAV410    (Stage 2 — thermal domain shift)", uav410)

    print("\n[3/3] CST Anti-UAV (Stage 3) ...")
    cst   = collect_cst(args.cst_root, args.splits)
    print_table("CST Anti-UAV   (Stage 3 — extreme tiny targets)", cst)

    datasets = [
        ("Anti-UAV-RGBT\n(Stage 1)", rgbt),
        ("Anti-UAV