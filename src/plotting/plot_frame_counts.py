#!/usr/bin/env python3
"""
plot_frame_counts.py
--------------------
Regenerates the appendix dataset figures, reusing the per-format parsing in
audit_datasets.py. Counts include all splits (train/val/test) for every
dataset; audit_datasets.PATHS was previously limited to the Anti-UAV-RGBT
test split, which undercounted that dataset.

Plots ANNOTATED frames (exist == 1) per split, matching the annotated-frame
counts reported in the thesis.

Outputs:
  <out-dir>/fig_frame_counts.png   annotated frames per dataset split
  <out-dir>/fig_visibility.png     visible fraction (exist=1 / total) per split
  Console: per-dataset totals (all splits) and train+val subtotals.

Usage (CPU only):
  python plot_frame_counts.py --out-dir <output-dir>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_datasets import (
    PATHS,
    audit_rgbt_dataset,
    audit_json_dataset,
    audit_ard100,
    audit_cst,
)

# Consistent per-dataset colours (match the original figure legend)
DS_COLOR = {
    "Anti-UAV-RGBT": "#1f77b4",
    "Anti-UAV410":   "#ff7f0e",
    "ARD100":        "#9467bd",
    "CST Anti-UAV":  "#d62728",
}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2, "all": 3}


def collect():
    """Return list of (dataset, split, frames, annotated), preserving split order.

    Prints per-dataset progress; parsing all annotations takes a few minutes.
    """
    rows = []

    def add(ds, audit_call):
        print(f"[auditing] {ds} ... (reading annotations)", flush=True)
        results = audit_call()
        for split in sorted(results, key=lambda s: SPLIT_ORDER.get(s, 9)):
            s = results[split]
            ann = s.get("annotated", 0)
            rows.append((ds, split, s.get("frames", 0), ann))
            print(f"    {ds:16} {split:5}: {ann:,} annotated "
                  f"/ {s.get('frames', 0):,} frames", flush=True)

    add("Anti-UAV-RGBT", lambda: audit_rgbt_dataset(PATHS["Anti-UAV-RGBT"]))
    add("Anti-UAV410",   lambda: audit_json_dataset(PATHS["Anti-UAV410"], ann_filename="IR_label.json"))
    add("ARD100",        lambda: audit_ard100(PATHS["ARD100"]["all"]))
    add("CST Anti-UAV",  lambda: audit_cst(PATHS["CST-AntiUAV"]))
    return rows


def _xlabels(rows):
    out = []
    for ds, split, _, _ in rows:
        out.append(ds if split == "all" else f"{ds}\n({split})")
    return out


def fig_frame_counts(rows, out_path):
    labels = _xlabels(rows)
    vals   = [ann / 1000.0 for _, _, _, ann in rows]      # thousands, annotated
    colors = [DS_COLOR[ds] for ds, _, _, _ in rows]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{v:.0f}k", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Annotated frames (thousands)", fontsize=14)
    ax.set_title("Annotated Frame Count per Dataset Split", fontsize=15, fontweight="bold")
    # de-duplicated legend by dataset
    seen = {}
    for ds in DS_COLOR:
        if any(r[0] == ds for r in rows):
            seen[ds] = plt.Rectangle((0, 0), 1, 1, color=DS_COLOR[ds])
    ax.legend(seen.values(), seen.keys(), fontsize=13, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


def fig_visibility(rows, out_path):
    labels = _xlabels(rows)
    vals = [100.0 * ann / fr if fr else 0.0 for _, _, fr, ann in rows]
    colors = [DS_COLOR[ds] for ds, _, _, _ in rows]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Visible frames (exist=1) %", fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_title("Target Visibility per Dataset Split", fontsize=15, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


def print_totals(rows):
    print("\n" + "=" * 60)
    print("ANNOTATED-FRAME TOTALS")
    print("=" * 60)
    by_ds = {}
    for ds, split, fr, ann in rows:
        d = by_ds.setdefault(ds, {"all": 0, "trainval": 0})
        d["all"] += ann
        if split in ("train", "val"):
            d["trainval"] += ann
    for ds, d in by_ds.items():
        tv = f", train+val={d['trainval']:,}" if d["trainval"] else ""
        print(f"  {ds:16}  all-splits={d['all']:,}{tv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=".", help="where to save the PNGs")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out.resolve()}", flush=True)
    print("Starting dataset audit ...", flush=True)
    rows = collect()
    fig_frame_counts(rows, out / "fig_frame_counts.png")
    fig_visibility(rows, out / "fig_visibility.png")
    print_totals(rows)


if __name__ == "__main__":
    main()
