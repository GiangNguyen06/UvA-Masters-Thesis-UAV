#!/usr/bin/env python3
"""
assemble_vis_comparison.py
--------------------------
Assemble the 3-panel Stage 1 / Stage 2 / Stage 3 detection comparison
(fig_vis_comparison.png) with LARGE, legible panel titles.

Inputs: three detection-visualisation images of the SAME Anti-UAV-RGBT val
frame, one per checkpoint. Produce them first with visualise_detections_rgbt.py
(run once per checkpoint so the predicted boxes are drawn on the frame), then
pass the three resulting PNGs here. This script only composes and labels them.

Usage:
  python assemble_vis_comparison.py \
      --stage1 s1_frame.png --stage2 s2_frame.png --stage3 s3_frame.png \
      --out /projects/prjs2041/analysis/fig_vis_comparison.png
  # then copy fig_vis_comparison.png into the Overleaf media/img/ folder.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DEFAULT_TITLES = [
    "Stage 1  (T1 ceiling)\nmAP@0.5 = 0.6725",
    "Stage 2  (after KD)\nmAP@0.5 = 0.640,  FM = $-$0.033",
    "Stage 3  (naive, ep 3)\nT1 mAP@0.5 = 0.068,  FM = $-$0.605",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", required=True, help="detection image, Stage 1 checkpoint")
    ap.add_argument("--stage2", required=True, help="detection image, Stage 2 checkpoint")
    ap.add_argument("--stage3", required=True, help="detection image, Stage 3 checkpoint")
    ap.add_argument("--out", default="fig_vis_comparison.png")
    ap.add_argument("--titles", nargs=3, default=DEFAULT_TITLES,
                    help="three panel titles (override the defaults)")
    ap.add_argument("--titlesize", type=float, default=15.0,
                    help="panel title font size (large, for legibility)")
    args = ap.parse_args()

    paths = [args.stage1, args.stage2, args.stage3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=200)
    for ax, p, t in zip(axes, paths, args.titles):
        ax.imshow(mpimg.imread(p))
        ax.set_title(t, fontsize=args.titlesize, fontweight="bold", pad=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
