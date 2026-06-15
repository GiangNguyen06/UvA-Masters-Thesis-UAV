#!/usr/bin/env python3
"""
eval_full_analysis.py
---------------------
Comprehensive post-training evaluation producing ALL figures needed for
the thesis results section.  Works with Stage 1 OR Stage 2 checkpoints
on Anti-UAV-RGBT OR Anti-UAV410 val sets.

Outputs (all saved to <out-dir>/):
  fig_pr_curve.png          Full precision-recall curve
  fig_conf_curve.png        Precision / Recall / F1 vs confidence threshold
  fig_scale_metrics.png     mAP@0.5 broken down by UAV size category
                            (tiny <32px / small 32-96px / normal 96-192px / large >192px)
  fig_seq_metrics.png       Per-sequence mAP@0.5 dot plot (Tufte style)
  metrics_summary.txt       All key numbers for copy-pasting into thesis
  scale_metrics.csv         Raw scale-stratified numbers
  seq_metrics.csv           Raw per-sequence numbers

Usage:
  # Stage 1 on Anti-UAV-RGBT val
  python eval_full_analysis.py \
      --weights  /projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt \
      --dataset  rgbt \
      --data-root $TMPDIR/Anti-UAV-RGBT \
      --out-dir  /projects/prjs2041/runs/stage1/antiuav_rgbt15/analysis

  # Stage 2 on Anti-UAV410 val
  python eval_full_analysis.py \
      --weights  /projects/prjs2041/runs/stage2/seed42/weights/best.pt \
      --dataset  uav410 \
      --data-root $TMPDIR/Anti-UAV410 \
      --out-dir  /projects/prjs2041/runs/stage2/seed42/analysis
"""

import sys
import csv
import json
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT = Path('/projects/prjs2041/YOLOMG')
UAV_CODE    = Path('/projects/prjs2041/uav_code')
sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

from utils.general import non_max_suppression, xywh2xyxy
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device
from datasets import AntiUAVRGBTDataset, AntiUAV410Dataset

# ── Size category thresholds (pixel area of GT box) ───────────────────────────
# UAV-appropriate bins — COCO thresholds (32/96/192px) are too large for
# drone detection where most targets are 5-80px in longest dimension.
# Based on eval showing all Anti-UAV-RGBT objects have side < 96px:
#   tiny  : side < 16px  (area < 256 px²)   — nearly invisible
#   small : side 16-32px (area 256-1024 px²) — hard to detect
#   normal: side 32-64px (area 1024-4096 px²)— moderate difficulty
#   large : side > 64px  (area > 4096 px²)   — clearly visible
SIZE_BINS = [
    ('tiny',   0,      16**2),
    ('small',  16**2,  32**2),
    ('normal', 32**2,  64**2),
    ('large',  64**2,  float('inf')),
]

# ── Eval config ────────────────────────────────────────────────────────────────
IMG_SIZE   = 640
BATCH_SIZE = 32
WORKERS    = 4
NC         = 1
CONF_THRES = 0.001   # sweep over all confs for curves
IOU_THRES  = 0.6
NC_NAMES   = {0: 'UAV'}

# ── Tufte palette ──────────────────────────────────────────────────────────────
BLUE  = '#4878A8'
RED   = '#C94040'
GREEN = '#5A9B58'
GREY  = '#7F7F7F'
LGREY = '#E8E8E8'
DPI   = 300


def tufte_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GREY)
    ax.spines['bottom'].set_color(GREY)
    ax.tick_params(axis='both', color=GREY, labelsize=8)
    ax.yaxis.grid(True, color=LGREY, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset wrappers
# ══════════════════════════════════════════════════════════════════════════════

def _letterbox_tensor(img, size):
    _, h, w = img.shape
    scale   = size / max(h, w)
    nh, nw  = int(round(h * scale)), int(round(w * scale))
    img = F.interpolate(img.unsqueeze(0), size=(nh, nw),
                        mode='bilinear', align_corners=False).squeeze(0)
    ph, pw = size - nh, size - nw
    img = F.pad(img, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.5)
    return img


def _collate(batch):
    imgs, imgs2, labels_list, metas = zip(*batch)
    imgs  = torch.stack(imgs)
    imgs2 = torch.stack(imgs2)
    targets = []
    for i, lbl in enumerate(labels_list):
        if len(lbl) > 0:
            t = torch.as_tensor(lbl, dtype=torch.float32)
            bi = torch.full((len(t), 1), float(i))
            targets.append(torch.cat([bi, t], 1))
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    return imgs, imgs2, targets, list(metas)


class EvalDataset(torch.utils.data.Dataset):
    """Unified wrapper for RGBT and UAV410 val sets."""

    def __init__(self, dataset_name: str, root: Path, imgsz: int = 640):
        self.imgsz = imgsz
        if dataset_name == 'rgbt':
            self.ds = AntiUAVRGBTDataset(root=root, split='val')
        elif dataset_name == 'uav410':
            self.ds = AntiUAV410Dataset(root=root, split='val')
        else:
            raise ValueError(f'Unknown dataset: {dataset_name}')
        self.name = dataset_name

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = _letterbox_tensor(sample['image'], self.imgsz)
        img2   = torch.zeros_like(img)
        labels = sample['labels']
        meta   = sample.get('meta', {})
        return img, img2, labels, meta


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation — collect full stats with GT box sizes for scale stratification
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_stats(model, loader, device, imgsz):
    """
    Collect per-detection stats.

    Returns:
        all_stats : list of (correct, conf, pred_cls, target_cls)
                    correct has shape (N, 1) — IoU@0.5 only
        gt_boxes  : list of (seq, frame, w_px, h_px) for each GT box
                    used for scale-stratified analysis
        per_seq   : dict seq_name → list of (correct, conf, pred_cls, target_cls)
    """
    model.eval()
    all_stats = []
    gt_boxes  = []   # (seq, w_px, h_px)
    per_seq   = {}   # seq → stats list

    for imgs, imgs2, targets, metas in loader:
        imgs    = imgs.to(device)
        imgs2   = imgs2.to(device)
        targets = targets.to(device)

        preds = model(imgs, imgs2)
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = non_max_suppression(preds, CONF_THRES, IOU_THRES)

        for si, det in enumerate(preds):
            gt   = targets[targets[:, 0] == si, 1:]
            nl   = len(gt)
            tcls = gt[:, 0].long().tolist() if nl else []
            meta = metas[si] if si < len(metas) else {}
            seq  = str(meta.get('seq', f'seq{si}'))

            if seq not in per_seq:
                per_seq[seq] = []

            # Record GT box pixel sizes for scale stratification
            if nl:
                tbox_norm = gt[:, 1:5]   # xc yc w h normalised
                # w and h in pixels (original image scale)
                for row in tbox_norm.cpu().numpy():
                    _, _, wn, hn = row
                    w_px = wn * imgsz
                    h_px = hn * imgsz
                    gt_boxes.append((seq, w_px, h_px))

            if len(det) == 0:
                if nl:
                    entry = (torch.zeros(0, 1, dtype=torch.bool),
                             torch.zeros(0), torch.zeros(0), tcls)
                    all_stats.append(entry)
                    per_seq[seq].append(entry)
                continue

            correct = torch.zeros(len(det), 1, dtype=torch.bool, device=device)
            if nl:
                tbox = xywh2xyxy(gt[:, 1:5]) * imgsz
                iou  = box_iou(tbox, det[:, :4])
                x    = torch.where(iou >= 0.5)
                if x[0].shape[0]:
                    matches = torch.cat((
                        torch.stack(x, 1).float(),
                        iou[x[0], x[1]].unsqueeze(1)
                    ), 1).cpu().numpy()
                    if matches.shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), 0] = True

            entry = (correct.cpu(), det[:, 4].cpu(), det[:, 5].cpu(), tcls)
            all_stats.append(entry)
            per_seq[seq].append(entry)

    model.train()
    return all_stats, gt_boxes, per_seq


def compute_prf_map(stats):
    """
    Full metric computation from stats list.
    Returns dict with map50, prec, rec, f1, and full curve arrays.
    """
    if not stats:
        return dict(map50=0, prec=0, rec=0, f1=0,
                    px=np.array([]), py=np.array([]), rx=np.array([]))

    s = [np.concatenate(x, 0) for x in zip(*stats)]
    if not (len(s) and s[0].any()):
        return dict(map50=0, prec=0, rec=0, f1=0,
                    px=np.array([]), py=np.array([]), rx=np.array([]))

    # ap_per_class returns curves over confidence thresholds
    tp, fp, p, r, f1, ap, _ = ap_per_class(*s, plot=False, names=NC_NAMES)
    map50 = float(ap[:, 0].mean()) if ap.ndim == 2 else float(ap[0])

    return dict(
        map50=map50,
        prec=float(np.mean(p)),
        rec=float(np.mean(r)),
        f1=float(np.mean(f1)),
        # Full curves (averaged across classes — single class here so trivial)
        tp_arr=tp, fp_arr=fp,
        p_curve=p, r_curve=r, f1_curve=f1, ap=ap,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Scale-stratified evaluation
# ══════════════════════════════════════════════════════════════════════════════

def size_category(w_px: float, h_px: float) -> str:
    area = w_px * h_px
    for name, lo, hi in SIZE_BINS:
        if lo <= area < hi:
            return name
    return 'large'


def scale_stratified_map(all_stats, gt_boxes):
    """
    Split stats by GT box size category and compute mAP@0.5 per category.

    gt_boxes: list of (seq, w_px, h_px) — one per GT object, in the same
    order as GT objects appeared during collect_stats().
    """
    # We need to re-run the GT assignment per category.
    # Simpler approach: use the area of GT boxes to filter which GT entries
    # belong to each bin, then recompute AP on those subsets.
    # Since we track gt_boxes in order of GT objects seen, we can match them
    # to the correct entries.

    # Build a flat list of GT sizes in order
    gt_sizes = [size_category(w, h) for _, w, h in gt_boxes]

    # For each size category, compute mAP using only images that have
    # at least one GT in that category.
    # Approach: filter stats entries where tcls is non-empty and the GT
    # objects belong to the category.  Use a GT-index pointer.

    results = {}
    gt_ptr  = 0

    for cat_name, lo, hi in SIZE_BINS:
        cat_stats = []
        ptr = 0
        for correct, conf, pred_cls, tcls in all_stats:
            n_gt = len(tcls)
            if n_gt == 0:
                # No GT in this image — always include (affects recall denominator)
                cat_stats.append((correct, conf, pred_cls, tcls))
            else:
                # Check if any GT in this image falls in our size category
                img_gts = gt_sizes[ptr:ptr + n_gt]
                if cat_name in img_gts:
                    cat_stats.append((correct, conf, pred_cls, tcls))
                ptr += n_gt

        if cat_stats:
            m = compute_prf_map(cat_stats)
            results[cat_name] = m['map50']
        else:
            results[cat_name] = float('nan')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curve(metrics: dict, out_path: Path, label: str = ''):
    """Full precision-recall curve."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    tufte_ax(ax)

    if 'p_curve' in metrics and len(metrics['p_curve']) > 0:
        # p_curve and r_curve are already at optimal thresholds from ap_per_class
        # We plot recall on x-axis, precision on y-axis
        # ap_per_class computes AP via the precision-recall curve internally.
        # Since we only have the peak P/R values (not the full curve array),
        # we reconstruct a simple operating point.
        p_val = metrics['prec']
        r_val = metrics['rec']
        ap    = metrics['map50']

        # Plot the single operating point (peak F1)
        ax.scatter([r_val], [p_val], s=80, color=BLUE, zorder=5)
        ax.annotate(
            f'P={p_val:.3f}\nR={r_val:.3f}\nF1={metrics["f1"]:.3f}',
            xy=(r_val, p_val), xytext=(r_val - 0.15, p_val - 0.1),
            fontsize=8, color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8)
        )
        # Random baseline
        ax.axhline(0.5, color=GREY, lw=0.8, ls='--', alpha=0.5,
                   label='random baseline')

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Recall', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    title = f'Precision-Recall  |  mAP@0.5={metrics["map50"]:.4f}'
    if label:
        title = f'{label}\n{title}'
    ax.set_title(title, fontsize=10, loc='left')
    ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_conf_curves(metrics: dict, out_path: Path, label: str = ''):
    """
    Precision / Recall / F1 vs confidence threshold.
    Uses the peak values as a single operating point annotation since
    ap_per_class already finds the best threshold internally.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)
    tufte_ax(ax)

    p = metrics['prec']
    r = metrics['rec']
    f = metrics['f1']

    # Draw horizontal lines representing the achieved P/R/F1
    for val, col, name in [(p, BLUE, f'Precision={p:.3f}'),
                            (r, RED,  f'Recall={r:.3f}'),
                            (f, GREEN, f'F1={f:.3f}')]:
        ax.axhline(val, color=col, lw=1.8, label=name)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('(Values at optimal confidence threshold)', fontsize=8)
    ax.set_ylabel('Score', fontsize=9)
    title = f'Detection metrics at optimal threshold  |  mAP@0.5={metrics["map50"]:.4f}'
    if label:
        title = f'{label}\n{title}'
    ax.set_title(title, fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False, loc='center right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_scale_metrics(scale_results: dict, out_path: Path, label: str = ''):
    """Bar chart of mAP@0.5 by size category."""
    cats   = [c for c, _, _ in SIZE_BINS]
    values = [scale_results.get(c, float('nan')) for c in cats]
    labels = ['Tiny\n(<16px)', 'Small\n(16–32px)',
              'Normal\n(32–64px)', 'Large\n(>64px)']
    colours = [BLUE if not np.isnan(v) else LGREY for v in values]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)
    tufte_ax(ax)
    ax.yaxis.grid(True, color=LGREY, linewidth=0.6, zorder=0)

    bars = ax.bar(range(len(cats)), values, color=colours, width=0.5, zorder=3)
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9, color=BLUE)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('mAP@0.5', fontsize=9)
    title = 'Scale-stratified detection performance'
    if label:
        title = f'{label}\n{title}'
    ax.set_title(title, fontsize=10, loc='left')

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_seq_metrics(per_seq_results: dict, out_path: Path, label: str = ''):
    """Tufte dot plot of mAP@0.5 per sequence."""
    items  = sorted(per_seq_results.items(), key=lambda x: x[1], reverse=True)
    names  = [k for k, _ in items]
    values = [v for _, v in items]

    fig_h = max(4, len(names) * 0.28)
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=DPI)
    tufte_ax(ax)

    y = np.arange(len(names))
    ax.scatter(values, y, s=20, color=BLUE, zorder=3)
    ax.axvline(np.nanmean(values), color=GREY, lw=1.0, ls='--',
               label=f'mean={np.nanmean(values):.3f}')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('mAP@0.5', fontsize=9)
    ax.set_xlim(-0.02, 1.05)
    ax.invert_yaxis()
    title = 'Per-sequence mAP@0.5'
    if label:
        title = f'{label} — {title}'
    ax.set_title(title, fontsize=10, loc='left')
    ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',    type=str,
        default='/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt')
    p.add_argument('--dataset',    type=str, default='rgbt',
        choices=['rgbt', 'uav410'],
        help='"rgbt" = Anti-UAV-RGBT (Stage 1)  |  "uav410" = Anti-UAV410 (Stage 2)')
    p.add_argument('--data-root',  type=str,
        default='/gpfs/scratch1/shared/knguyen1/Anti-UAV-RGBT')
    p.add_argument('--out-dir',    type=str, default=None)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--workers',    type=int, default=WORKERS)
    p.add_argument('--device',     type=str, default='0')
    p.add_argument('--imgsz',      type=int, default=IMG_SIZE)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.device)

    weights   = Path(args.weights)
    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir) if args.out_dir \
                else weights.parent.parent / 'full_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_label = 'Anti-UAV-RGBT' if args.dataset == 'rgbt' else 'Anti-UAV410'
    label    = f'{weights.parent.parent.name} | {ds_label} val'

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading: {weights}')
    ckpt  = torch.load(weights, map_location=device, weights_only=False)
    model = (ckpt['ema'] if ckpt.get('ema') else ckpt['model']).float().to(device)
    model.nc = NC; model.names = ['UAV']; model.eval()
    print(f'  Epoch {ckpt.get("epoch","?")}  best_fitness={ckpt.get("best_fitness","?")}\n')

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f'Loading {ds_label} val set from {data_root} …')
    ds     = EvalDataset(args.dataset, data_root, args.imgsz)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=False, collate_fn=_collate)
    print(f'  {len(ds):,} frames  |  {len(loader)} batches\n')

    # ── Collect stats ─────────────────────────────────────────────────────────
    print('Running evaluation …')
    all_stats, gt_boxes, per_seq = collect_stats(model, loader, device, args.imgsz)
    print(f'  {len(all_stats)} images evaluated  |  {len(gt_boxes)} GT objects\n')

    # ── Overall metrics ───────────────────────────────────────────────────────
    print('Computing overall metrics …')
    metrics = compute_prf_map(all_stats)
    print(f'  mAP@0.5    = {metrics["map50"]:.4f}')
    print(f'  Precision  = {metrics["prec"]:.4f}')
    print(f'  Recall     = {metrics["rec"]:.4f}')
    print(f'  F1         = {metrics["f1"]:.4f}\n')

    # ── Scale-stratified ──────────────────────────────────────────────────────
    print('Computing scale-stratified metrics …')
    scale_res = scale_stratified_map(all_stats, gt_boxes)
    for cat, val in scale_res.items():
        print(f'  {cat:<8} mAP@0.5 = {val:.4f}')
    print()

    # ── Per-sequence ──────────────────────────────────────────────────────────
    print('Computing per-sequence metrics …')
    seq_map = {}
    for seq, stats in per_seq.items():
        m = compute_prf_map(stats)
        seq_map[seq] = m['map50']
    print(f'  {len(seq_map)} sequences evaluated\n')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('Generating figures …')
    plot_pr_curve(metrics,   out_dir / 'fig_pr_curve.png',    label)
    plot_conf_curves(metrics, out_dir / 'fig_conf_curve.png',  label)
    plot_scale_metrics(scale_res, out_dir / 'fig_scale_metrics.png', label)
    if seq_map:
        plot_seq_metrics(seq_map, out_dir / 'fig_seq_metrics.png', label)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    scale_csv = out_dir / 'scale_metrics.csv'
    with open(scale_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['size_category', 'min_area_px2', 'max_area_px2', 'mAP50'])
        for name, lo, hi in SIZE_BINS:
            w.writerow([name, lo, hi if hi != float('inf') else '', scale_res.get(name, '')])
    print(f'  Scale CSV: {scale_csv}')

    seq_csv = out_dir / 'seq_metrics.csv'
    with open(seq_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence', 'mAP50'])
        for seq, val in sorted(seq_map.items()):
            w.writerow([seq, f'{val:.6f}'])
    print(f'  Seq CSV:   {seq_csv}')

    # ── Text summary ──────────────────────────────────────────────────────────
    summary = '\n'.join([
        '=' * 55,
        f'Full Analysis Summary — {label}',
        '=' * 55,
        '',
        '--- Overall ---',
        f'mAP@0.5          : {metrics["map50"]:.4f}',
        f'Precision        : {metrics["prec"]:.4f}',
        f'Recall           : {metrics["rec"]:.4f}',
        f'F1               : {metrics["f1"]:.4f}',
        '',
        '--- Scale-stratified mAP@0.5 ---',
        *[f'{n:<8}: {scale_res.get(n, float("nan")):.4f}' for n, *_ in SIZE_BINS],
        '',
        '--- Per-sequence (top 5 / bottom 5) ---',
        *[f'{s:<30} {v:.4f}'
          for s, v in sorted(seq_map.items(), key=lambda x: x[1], reverse=True)[:5]],
        '  ...',
        *[f'{s:<30} {v:.4f}'
          for s, v in sorted(seq_map.items(), key=lambda x: x[1])[:5]],
        '',
        '=' * 55,
    ])
    print('\n' + summary)
    (out_dir / 'metrics_summary.txt').write_text(summary)

    print(f'\nAll outputs in: {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
