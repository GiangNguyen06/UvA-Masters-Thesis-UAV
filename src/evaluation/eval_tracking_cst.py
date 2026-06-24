#!/usr/bin/env python3
"""
eval_tracking_cst.py
--------------------
Single-object tracking evaluation on CST Anti-UAV val sequences.

For each frame in each val sequence, runs inference with YOLOMG (img2=zeros)
and matches the highest-confidence detection to the ground-truth drone box.

Metrics (per sequence and aggregate):
  SR@0.5   Success Rate at IoU ≥ 0.5  — fraction of GT frames detected
  AUC-SR   Area under Success Plot (IoU thresholds 0.0–1.0, step 0.05)
  PR@20    Precision at centre-distance ≤ 20 px
  PR@10    Precision at centre-distance ≤ 10 px  (tighter; useful for tiny targets)
  AUC-PR   Area under Precision Plot (dist thresholds 0–50 px, step 1 px)
  IDSW     ID Switches: times tracking re-acquires target after losing it
           (IoU < 0.1 → lost; IoU ≥ 0.5 following a lost state → switch)

Outputs in <out-dir>/:
  fig_success_plot.png    Success Rate vs IoU threshold (AUC shaded)
  fig_precision_plot.png  Precision vs centre-distance threshold
  fig_seq_sr.png          Per-sequence SR@0.5 dot plot
  tracking_summary.txt    All key numbers
  seq_results.csv         Per-sequence table

Usage:
  python eval_tracking_cst.py \\
      --weights  /projects/prjs2041/runs/stage3/naive2/weights/best.pt \\
      --cst-root /projects/prjs2041/datasets/CST-AntiUAV \\
      --out-dir  /projects/prjs2041/analysis/tracking_eval \\
      --device   0
"""

import sys
import csv
import math
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT = Path('/projects/prjs2041/YOLOMG')
UAV_CODE    = Path('/projects/prjs2041/uav_code')
sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

from utils.general import non_max_suppression
from utils.torch_utils import select_device
from datasets.cst import CSTDataset

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 640
CONF_THRES  = 0.10   # lower threshold than detection eval — track through low-conf frames
IOU_THRES   = 0.45
NC          = 1

# Success plot: IoU thresholds
SR_THRESHOLDS = np.arange(0.0, 1.05, 0.05)

# Precision plot: centre-distance thresholds (pixels in original image space)
PR_THRESHOLDS = np.arange(0, 51, 1)

# Track loss/reacq thresholds
TRACK_LOSS_IOU  = 0.10   # IoU below this → track lost
TRACK_FOUND_IOU = 0.50   # IoU at or above this → track re-acquired

# ── Palette ────────────────────────────────────────────────────────────────────
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
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def letterbox(img_bgr, size=640):
    h, w   = img_bgr.shape[:2]
    scale  = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img    = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = size - nh, size - nw
    img    = cv2.copyMakeBorder(img, ph // 2, ph - ph // 2,
                                 pw // 2, pw - pw // 2,
                                 cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img, scale, pw // 2, ph // 2    # img, scale, pad_l, pad_t


def unletterbox(x1, y1, x2, y2, scale, pad_l, pad_t, orig_w, orig_h):
    x1 = max(0, int((x1 - pad_l) / scale))
    y1 = max(0, int((y1 - pad_t) / scale))
    x2 = min(orig_w, int((x2 - pad_l) / scale))
    y2 = min(orig_h, int((y2 - pad_t) / scale))
    return x1, y1, x2, y2


@torch.no_grad()
def detect_frame(model, img_bgr, device):
    """
    Returns list of (x1, y1, x2, y2, conf) in original image pixel coords,
    sorted by confidence descending.  Empty list if no detections.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    lb, scale, pad_l, pad_t = letterbox(img_bgr, IMG_SIZE)
    rgb  = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    img1 = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    img2 = torch.zeros_like(img1)

    out = model(img1, img2)
    if isinstance(out, tuple):
        out = out[0]
    dets = non_max_suppression(out, CONF_THRES, IOU_THRES)[0]

    results = []
    if dets is not None and len(dets):
        for *xyxy, conf, _ in dets.cpu().numpy():
            x1, y1, x2, y2 = unletterbox(
                xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                scale, pad_l, pad_t, orig_w, orig_h)
            results.append((x1, y1, x2, y2, float(conf)))
        results.sort(key=lambda r: -r[4])   # highest conf first
    return results


# ══════════════════════════════════════════════════════════════════════════════
# IoU and centre-distance
# ══════════════════════════════════════════════════════════════════════════════

def iou_xyxy(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ba = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def centre_dist(pred_xyxy, gt_xyxy):
    """Euclidean distance between box centres (pixels)."""
    px = (pred_xyxy[0] + pred_xyxy[2]) / 2
    py = (pred_xyxy[1] + pred_xyxy[3]) / 2
    gx = (gt_xyxy[0]  + gt_xyxy[2])  / 2
    gy = (gt_xyxy[1]  + gt_xyxy[3])  / 2
    return math.sqrt((px - gx) ** 2 + (py - gy) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# Per-sequence evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_sequence(model, seq_entries, device):
    """
    Evaluate one sequence.

    seq_entries: list of CSTDataset index entries (same seq, ordered by frame).
    Returns dict with per-frame arrays and aggregate counts.
    """
    iou_list   = []   # IoU per GT frame (exist==1)
    dist_list  = []   # centre dist per GT frame (exist==1, detection present)
    idsw       = 0
    track_lost = False   # True after IoU drops below TRACK_LOSS_IOU

    for entry in seq_entries:
        img_path = entry.get('img_path')
        if img_path is None or not Path(img_path).exists():
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        exist = entry['exist']
        box   = entry['box']   # [x, y, w, h] pixel, top-left origin

        # Build GT xyxy in pixel space
        if exist == 1 and box is not None and len(box) == 4:
            x, y, w, h = box
            if w > 0 and h > 0:
                gt_xyxy = (x, y, x + w, y + h)
            else:
                continue   # zero-size box → skip
        else:
            continue   # no drone in this frame — skip (don't penalise in SR/PR)

        # Inference
        dets = detect_frame(model, img_bgr, device)

        if not dets:
            iou_list.append(0.0)
            # Track stays in lost state; no IDSW (can't re-acquire with no det)
            track_lost = True
            continue

        best = dets[0]   # highest confidence detection
        pred_xyxy = best[:4]

        iou  = iou_xyxy(pred_xyxy, gt_xyxy)
        dist = centre_dist(pred_xyxy, gt_xyxy)
        iou_list.append(iou)
        dist_list.append(dist)

        # IDSW tracking logic
        if iou >= TRACK_FOUND_IOU:
            if track_lost:
                idsw      += 1
                track_lost = False
        elif iou < TRACK_LOSS_IOU:
            track_lost = True
        # Between TRACK_LOSS_IOU and TRACK_FOUND_IOU: ambiguous — keep current state

    n_gt = len(iou_list)
    if n_gt == 0:
        return None

    iou_arr  = np.array(iou_list,  dtype=np.float32)
    dist_arr = np.array(dist_list, dtype=np.float32)

    # Success Rate at each threshold
    sr_curve = np.array([float(np.mean(iou_arr >= t)) for t in SR_THRESHOLDS])
    auc_sr   = float(np.trapz(sr_curve, SR_THRESHOLDS) / (SR_THRESHOLDS[-1] - SR_THRESHOLDS[0]))

    # Precision curve (only frames with a detection have a dist value)
    n_with_det = len(dist_arr)
    if n_with_det > 0:
        pr_curve = np.array([float(np.mean(dist_arr <= t)) * (n_with_det / n_gt)
                             for t in PR_THRESHOLDS])
    else:
        pr_curve = np.zeros(len(PR_THRESHOLDS), dtype=np.float32)

    auc_pr   = float(np.trapz(pr_curve, PR_THRESHOLDS) / PR_THRESHOLDS[-1]) if PR_THRESHOLDS[-1] > 0 else 0.0
    sr_at_05 = float(np.mean(iou_arr >= 0.5))
    pr_at_20 = float(pr_curve[np.searchsorted(PR_THRESHOLDS, 20)])
    pr_at_10 = float(pr_curve[np.searchsorted(PR_THRESHOLDS, 10)])

    return {
        'n_gt':     n_gt,
        'sr_at_05': sr_at_05,
        'auc_sr':   auc_sr,
        'pr_at_20': pr_at_20,
        'pr_at_10': pr_at_10,
        'auc_pr':   auc_pr,
        'idsw':     idsw,
        'sr_curve': sr_curve,
        'pr_curve': pr_curve,
        'mean_iou': float(np.mean(iou_arr)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_success(sr_curve_mean, auc_sr, out_path, label=''):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    tufte_ax(ax)
    ax.fill_between(SR_THRESHOLDS, sr_curve_mean, alpha=0.12, color=BLUE)
    ax.plot(SR_THRESHOLDS, sr_curve_mean, color=BLUE, lw=1.8,
            label=f'AUC={auc_sr:.3f}')
    ax.axvline(0.5, color=GREY, lw=0.8, ls='--', alpha=0.6)
    sr_05 = float(np.interp(0.5, SR_THRESHOLDS, sr_curve_mean))
    ax.scatter([0.5], [sr_05], color=BLUE, s=40, zorder=5)
    ax.annotate(f'SR@0.5={sr_05:.3f}', xy=(0.5, sr_05),
                xytext=(0.55, sr_05 + 0.05), fontsize=8, color=BLUE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('IoU threshold', fontsize=9)
    ax.set_ylabel('Success Rate', fontsize=9)
    title = 'Success Plot (SOT)'
    if label:
        title = f'{label}\n{title}'
    ax.set_title(title, fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_precision(pr_curve_mean, auc_pr, pr_at_20, out_path, label=''):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
    tufte_ax(ax)
    ax.fill_between(PR_THRESHOLDS, pr_curve_mean, alpha=0.12, color=GREEN)
    ax.plot(PR_THRESHOLDS, pr_curve_mean, color=GREEN, lw=1.8,
            label=f'AUC={auc_pr:.3f}')
    ax.axvline(20, color=GREY, lw=0.8, ls='--', alpha=0.6)
    ax.scatter([20], [pr_at_20], color=GREEN, s=40, zorder=5)
    ax.annotate(f'PR@20={pr_at_20:.3f}', xy=(20, pr_at_20),
                xytext=(22, pr_at_20 + 0.03), fontsize=8, color=GREEN)
    ax.set_xlim(0, 50); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Centre distance threshold (px)', fontsize=9)
    ax.set_ylabel('Precision', fontsize=9)
    title = 'Precision Plot (SOT)'
    if label:
        title = f'{label}\n{title}'
    ax.set_title(title, fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_seq_sr(seq_names, seq_sr, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(seq_names) * 0.3), 4), dpi=DPI)
    tufte_ax(ax)
    xs = np.arange(len(seq_names))
    ax.scatter(xs, seq_sr, color=BLUE, s=20, zorder=5)
    ax.axhline(np.mean(seq_sr), color=RED, lw=1.0, ls='--',
               label=f'mean={np.mean(seq_sr):.3f}')
    ax.set_xticks(xs)
    ax.set_xticklabels(seq_names, rotation=90, fontsize=5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('SR@0.5', fontsize=9)
    ax.set_title('Per-sequence Success Rate (SR@0.5)', fontsize=10, loc='left')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',  type=str,
        default='/projects/prjs2041/runs/stage3/naive2/weights/best.pt')
    p.add_argument('--cst-root', type=str,
        default='/projects/prjs2041/datasets/CST-AntiUAV')
    p.add_argument('--split',    type=str, default='val')
    p.add_argument('--out-dir',  type=str, default=None)
    p.add_argument('--device',   type=str, default='0')
    p.add_argument('--n-seqs',   type=int, default=0,
        help='Limit to first N sequences (0 = all)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.device)

    weights  = Path(args.weights)
    cst_root = Path(args.cst_root)
    out_dir  = Path(args.out_dir) if args.out_dir \
               else weights.parent.parent / 'tracking_eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f'{weights.parent.parent.name} | CST Anti-UAV {args.split}'

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading: {weights}')
    ckpt  = torch.load(weights, map_location=device, weights_only=False)
    model = (ckpt['ema'] if ckpt.get('ema') else ckpt['model']).float().to(device)
    model.nc = NC; model.names = ['UAV']; model.eval()
    print(f'  Epoch {ckpt.get("epoch","?")}  best_fitness={ckpt.get("best_fitness","?")}\n')

    # ── Build per-sequence index from CSTDataset ──────────────────────────────
    print(f'Indexing CST Anti-UAV {args.split} from {cst_root} …')
    ds = CSTDataset(root=cst_root, split=args.split, skip_empty=False)

    # Group entries by sequence
    from collections import OrderedDict
    seqs = OrderedDict()
    for entry in ds._index:
        s = entry['seq']
        if s not in seqs:
            seqs[s] = []
        seqs[s].append(entry)

    seq_names = list(seqs.keys())
    if args.n_seqs > 0:
        seq_names = seq_names[:args.n_seqs]
    print(f'  {len(seq_names)} sequences  |  '
          f'{sum(len(seqs[s]) for s in seq_names):,} frames total\n')

    # ── Per-sequence evaluation ───────────────────────────────────────────────
    results = {}
    for i, sname in enumerate(seq_names):
        print(f'  [{i+1:3d}/{len(seq_names)}] {sname} …', end=' ', flush=True)
        r = eval_sequence(model, seqs[sname], device)
        if r is None:
            print('(no GT frames, skip)')
            continue
        results[sname] = r
        print(f'SR@0.5={r["sr_at_05"]:.3f}  PR@20={r["pr_at_20"]:.3f}  '
              f'IDSW={r["idsw"]}  GT={r["n_gt"]}')

    if not results:
        print('No results — check dataset path.')
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_total  = sum(r['n_gt']   for r in results.values())
    sr_curve_mean = np.average(
        [r['sr_curve'] for r in results.values()],
        weights=[r['n_gt'] for r in results.values()], axis=0)
    pr_curve_mean = np.average(
        [r['pr_curve'] for r in results.values()],
        weights=[r['n_gt'] for r in results.values()], axis=0)

    agg_sr_05 = float(np.average([r['sr_at_05'] for r in results.values()],
                                  weights=[r['n_gt'] for r in results.values()]))
    agg_auc_sr = float(np.trapz(sr_curve_mean, SR_THRESHOLDS) /
                        (SR_THRESHOLDS[-1] - SR_THRESHOLDS[0]))
    agg_pr_20  = float(pr_curve_mean[np.searchsorted(PR_THRESHOLDS, 20)])
    agg_pr_10  = float(pr_curve_mean[np.searchsorted(PR_THRESHOLDS, 10)])
    agg_auc_pr = float(np.trapz(pr_curve_mean, PR_THRESHOLDS) / PR_THRESHOLDS[-1])
    total_idsw = sum(r['idsw'] for r in results.values())
    mean_iou   = float(np.average([r['mean_iou'] for r in results.values()],
                                   weights=[r['n_gt'] for r in results.values()]))

    print(f'\n{"="*55}')
    print(f'Tracking Evaluation Summary — {label}')
    print(f'{"="*55}')
    print(f'Sequences evaluated : {len(results)}')
    print(f'Total GT frames     : {n_total:,}')
    print(f'')
    print(f'SR@0.5  (Success Rate IoU≥0.5) : {agg_sr_05:.4f}')
    print(f'AUC-SR  (Success Plot AUC)     : {agg_auc_sr:.4f}')
    print(f'PR@20   (Precision ≤20px)      : {agg_pr_20:.4f}')
    print(f'PR@10   (Precision ≤10px)      : {agg_pr_10:.4f}')
    print(f'AUC-PR  (Precision Plot AUC)   : {agg_auc_pr:.4f}')
    print(f'Mean IoU (all GT frames)       : {mean_iou:.4f}')
    print(f'Total IDSW                     : {total_idsw}')
    print(f'{"="*55}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating figures …')
    plot_success(sr_curve_mean, agg_auc_sr, out_dir / 'fig_success_plot.png', label)
    plot_precision(pr_curve_mean, agg_auc_pr, agg_pr_20,
                   out_dir / 'fig_precision_plot.png', label)
    plot_seq_sr(list(results.keys()),
                [results[s]['sr_at_05'] for s in results],
                out_dir / 'fig_seq_sr.png')

    # ── Summary text ──────────────────────────────────────────────────────────
    summary = [
        f'Tracking Evaluation — {label}',
        f'Checkpoint : {weights}',
        f'Split      : {args.split}',
        f'',
        f'Sequences evaluated : {len(results)}',
        f'Total GT frames     : {n_total:,}',
        f'',
        f'SR@0.5  (Success Rate IoU≥0.5) : {agg_sr_05:.4f}',
        f'AUC-SR  (Success Plot AUC)     : {agg_auc_sr:.4f}',
        f'PR@20   (Precision ≤20px)      : {agg_pr_20:.4f}',
        f'PR@10   (Precision ≤10px)      : {agg_pr_10:.4f}',
        f'AUC-PR  (Precision Plot AUC)   : {agg_auc_pr:.4f}',
        f'Mean IoU (all GT frames)       : {mean_iou:.4f}',
        f'Total IDSW                     : {total_idsw}',
        f'',
        f'Note: IDSW counted as re-acquisitions after track loss.',
        f'  Track lost   : IoU < {TRACK_LOSS_IOU}',
        f'  Track re-acq : IoU ≥ {TRACK_FOUND_IOU} following a lost state',
    ]
    (out_dir / 'tracking_summary.txt').write_text('\n'.join(summary))
    print(f'  Saved: {out_dir}/tracking_summary.txt')

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / 'seq_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq', 'n_gt', 'sr_at_05', 'auc_sr',
                    'pr_at_20', 'pr_at_10', 'auc_pr', 'mean_iou', 'idsw'])
        for sname, r in results.items():
            w.writerow([sname, r['n_gt'], f'{r["sr_at_05"]:.4f}',
                        f'{r["auc_sr"]:.4f}', f'{r["pr_at_20"]:.4f}',
                        f'{r["pr_at_10"]:.4f}', f'{r["auc_pr"]:.4f}',
                        f'{r["mean_iou"]:.4f}', r['idsw']])
    print(f'  Saved: {csv_path}')

    print(f'\nAll outputs in: {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
