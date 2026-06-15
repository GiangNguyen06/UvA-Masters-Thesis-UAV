#!/usr/bin/env python3
"""
parameter_drift.py
------------------
Quantify how much the YOLOMG student model drifted from its Stage 1
initialisation after Stage 2 Teacher-Student UDA training.

For every named parameter shared between Stage 1 and Stage 2 checkpoints,
computes:
  - L2 norm of the weight difference  (||W2 - W1||₂ / ||W1||₂  relative)
  - Cosine similarity between flattened weight vectors
  - Element-wise absolute difference distribution (mean, std, max)

Produces three Tufte-style figures:
  1. fig_drift_l2.png   — per-layer relative L2 norm (dot plot, sorted)
  2. fig_drift_cos.png  — per-layer cosine similarity (dot plot, sorted)
  3. fig_drift_hist.png — distribution comparison of selected layer deltas

Usage (Snellius) — Stage 1 → Stage 2:
  python parameter_drift.py \
      --stage1  /projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt \
      --stage2  /projects/prjs2041/runs/stage2/stage2_uda1/weights/best.pt \
      --label-a "Stage 1" --label-b "Stage 2" \
      --out-dir /projects/prjs2041/runs/stage2/stage2_uda1/drift

Usage (Snellius) — Stage 2 → Stage 3 (naive):
  python parameter_drift.py \
      --stage1  /projects/prjs2041/runs/stage2/stage2_uda1/weights/best.pt \
      --stage2  /projects/prjs2041/runs/stage3/naive/weights/best.pt \
      --label-a "Stage 2" --label-b "Stage 3 (naive)" \
      --out-dir /projects/prjs2041/runs/stage3/naive/drift

Outputs:
  <out-dir>/fig_drift_l2.png
  <out-dir>/fig_drift_cos.png
  <out-dir>/fig_drift_hist.png
  <out-dir>/drift_stats.csv
"""

import sys
import argparse
import csv
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT = Path('/projects/prjs2041/YOLOMG')
UAV_CODE    = Path('/projects/prjs2041/uav_code')
sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

from utils.torch_utils import select_device


# ══════════════════════════════════════════════════════════════════════════════
# Tufte-style plot helpers
# ══════════════════════════════════════════════════════════════════════════════

MUTED_BLUE   = '#4878A8'
MUTED_RED    = '#C94040'
MUTED_GREY   = '#7F7F7F'
LIGHT_GREY   = '#E8E8E8'
BG           = 'white'

def tufte_ax(ax):
    """Apply Tufte spine / grid settings to an Axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MUTED_GREY)
    ax.spines['bottom'].set_color(MUTED_GREY)
    ax.tick_params(axis='both', color=MUTED_GREY, labelsize=8)
    ax.yaxis.grid(True, color=LIGHT_GREY, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
# Weight extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_state_dict(ckpt_path: Path, device) -> dict:
    """
    Load a checkpoint and return its state_dict.
    Prefers EMA weights; falls back to model weights.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'ema' in ckpt and ckpt['ema'] is not None:
        sd = ckpt['ema'].float().state_dict()
        source = 'EMA'
    else:
        sd = ckpt['model'].float().state_dict()
        source = 'model'
    epoch = ckpt.get('epoch', '?')
    print(f'  Loaded {ckpt_path.name}  [{source} weights, epoch={epoch}]')
    return sd


def layer_group(name: str) -> str:
    """
    Assign a named parameter to a coarse layer group for colouring.
    Groups: backbone0, backbone1, neck, head, other.
    """
    if name.startswith('model.0.') or name.startswith('model.1.') \
       or name.startswith('model.2.') or name.startswith('model.3.'):
        return 'backbone'
    if name.startswith('model.') and any(
        name.startswith(f'model.{i}.') for i in range(10, 23)
    ):
        return 'neck'
    if 'Detect' in name or name.startswith('model.24.') \
       or name.startswith('model.33.'):
        return 'head'
    return 'other'


GROUP_COLOURS = {
    'backbone': MUTED_BLUE,
    'neck':     '#7BAA58',
    'head':     MUTED_RED,
    'other':    MUTED_GREY,
}


# ══════════════════════════════════════════════════════════════════════════════
# Drift computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_drift(sd1: dict, sd2: dict) -> list[dict]:
    """
    Compute per-parameter drift statistics for all shared float parameters.
    Returns list of dicts with keys:
      name, group, n_params,
      l2_abs, l2_rel, cos_sim,
      diff_mean, diff_std, diff_max
    """
    records = []
    shared  = [k for k in sd1 if k in sd2
               and sd1[k].dtype in (torch.float32, torch.float16, torch.bfloat16)]

    for name in shared:
        w1 = sd1[name].float().cpu()
        w2 = sd2[name].float().cpu()

        if w1.shape != w2.shape:
            print(f'  [SKIP] shape mismatch: {name}  {w1.shape} vs {w2.shape}')
            continue

        delta   = (w2 - w1).flatten()
        w1_flat = w1.flatten()

        l2_abs = float(delta.norm())
        l2_w1  = float(w1_flat.norm())
        l2_rel = l2_abs / (l2_w1 + 1e-12)

        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                w1_flat.unsqueeze(0), w2.flatten().unsqueeze(0)
            )
        )

        diff_abs = delta.abs()
        records.append({
            'name':      name,
            'group':     layer_group(name),
            'n_params':  w1.numel(),
            'l2_abs':    l2_abs,
            'l2_rel':    l2_rel,
            'cos_sim':   cos_sim,
            'diff_mean': float(diff_abs.mean()),
            'diff_std':  float(diff_abs.std()),
            'diff_max':  float(diff_abs.max()),
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_dot(records: list[dict], key: str, xlabel: str, title: str,
             out_path: Path, highlight_top: int = 5):
    """
    Tufte-style horizontal dot plot of `key` for all layers,
    sorted descending, coloured by group.
    """
    recs   = sorted(records, key=lambda r: r[key], reverse=True)
    names  = [r['name'].replace('model.', 'm.') for r in recs]
    values = [r[key] for r in recs]
    groups = [r['group'] for r in recs]
    colours = [GROUP_COLOURS.get(g, MUTED_GREY) for g in groups]

    n = len(names)
    fig_h = max(4, n * 0.22)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
    tufte_ax(ax)

    y = np.arange(n)
    ax.scatter(values, y, c=colours, s=18, zorder=3, linewidths=0)

    # Range frame: trim spines to data range
    ax.spines['bottom'].set_bounds(min(values), max(values))
    ax.spines['left'].set_bounds(0, n - 1)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc='left', pad=6)
    ax.invert_yaxis()

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markersize=6, label=g)
        for g, c in GROUP_COLOURS.items()
    ]
    ax.legend(handles=handles, fontsize=7, loc='lower right',
              frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_hist(records: list[dict], sd1: dict, sd2: dict,
              out_path: Path, n_layers: int = 6,
              label_a: str = 'Stage 1', label_b: str = 'Stage 2'):
    """
    Plot weight distribution histograms for the n_layers with the largest
    relative L2 drift, comparing checkpoint A vs checkpoint B.
    """
    top = sorted(records, key=lambda r: r['l2_rel'], reverse=True)[:n_layers]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=150)
    axes = axes.flatten()

    for ax, rec in zip(axes, top):
        tufte_ax(ax)
        name = rec['name']
        w1 = sd1[name].float().cpu().flatten().numpy()
        w2 = sd2[name].float().cpu().flatten().numpy()

        bins = np.linspace(min(w1.min(), w2.min()),
                           max(w1.max(), w2.max()), 60)

        ax.hist(w1, bins=bins, alpha=0.55, color=MUTED_BLUE,
                label=label_a, density=True)
        ax.hist(w2, bins=bins, alpha=0.55, color=MUTED_RED,
                label=label_b, density=True)

        short = name.split('.')[-3:]
        ax.set_title('.'.join(short), fontsize=8, loc='left')
        ax.set_xlabel('weight value', fontsize=7)
        ax.set_ylabel('density', fontsize=7)
        ax.text(0.97, 0.95,
                f'Δrel={rec["l2_rel"]:.3f}\ncos={rec["cos_sim"]:.4f}',
                transform=ax.transAxes, fontsize=7, ha='right', va='top',
                color=MUTED_GREY)
        ax.legend(fontsize=6, frameon=False)

    # Hide any unused subplots
    for ax in axes[len(top):]:
        ax.set_visible(False)

    fig.suptitle(
        f'Weight distribution shift: {label_a} → {label_b}\n'
        f'(top {n_layers} layers by relative L2 drift)',
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_group_summary(records: list[dict], out_path: Path,
                       label_a: str = 'Stage 1', label_b: str = 'Stage 2'):
    """
    Box-and-dot summary of relative L2 drift per layer group.
    """
    groups = ['backbone', 'neck', 'head', 'other']
    data   = {g: [r['l2_rel'] for r in records if r['group'] == g]
              for g in groups}
    data   = {g: v for g, v in data.items() if v}  # drop empty

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    tufte_ax(ax)

    grp_labels = list(data.keys())
    colours    = [GROUP_COLOURS[g] for g in grp_labels]
    values     = [data[g] for g in grp_labels]

    positions = np.arange(len(grp_labels))
    bp = ax.boxplot(values, positions=positions, widths=0.35,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=2))

    for patch, c in zip(bp['boxes'], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # Jitter individual dots
    rng = np.random.default_rng(0)
    for xi, (vals, c) in enumerate(zip(values, colours)):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(xi + jitter, vals, s=12, color=c, alpha=0.7, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(grp_labels, fontsize=9)
    ax.set_ylabel('Relative L2 drift  ||W₂−W₁||₂ / ||W₁||₂', fontsize=9)
    ax.set_title(f'Parameter drift by layer group\n({label_a} → {label_b})',
                 fontsize=10, loc='left')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1',   type=str,
        default='/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt',
        help='Path to "before" checkpoint (Stage 1 best.pt for S1→S2; Stage 2 best.pt for S2→S3)')
    p.add_argument('--stage2',   type=str,
        default='/projects/prjs2041/runs/stage2/stage2_uda1/weights/best.pt',
        help='Path to "after" checkpoint')
    p.add_argument('--label-a',  type=str, default='Stage 1',
        help='Label for the "before" checkpoint in plot titles and legends')
    p.add_argument('--label-b',  type=str, default='Stage 2',
        help='Label for the "after" checkpoint in plot titles and legends')
    p.add_argument('--out-dir',  type=str, default=None,
        help='Output directory (default: stage2 weights dir parent/drift/)')
    p.add_argument('--device',   type=str, default='cpu',
        help='cpu is sufficient for this script (no GPU ops needed)')
    p.add_argument('--top-hist', type=int, default=6,
        help='Number of top-drift layers to show in histogram plot')
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.device)

    stage1_path = Path(args.stage1)
    stage2_path = Path(args.stage2)
    label_a     = args.label_a
    label_b     = args.label_b
    out_dir     = Path(args.out_dir) if args.out_dir \
                  else stage2_path.parent.parent / 'drift'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nParameter Drift Analysis')
    print(f'  {label_a} checkpoint : {stage1_path}')
    print(f'  {label_b} checkpoint : {stage2_path}')
    print(f'  Out dir              : {out_dir}\n')

    print('Loading checkpoints …')
    sd1 = extract_state_dict(stage1_path, device)
    sd2 = extract_state_dict(stage2_path, device)

    print(f'\nComputing drift for shared parameters …')
    records = compute_drift(sd1, sd2)
    print(f'  {len(records)} parameters analysed\n')

    # ── Global summary ────────────────────────────────────────────────────────
    l2_rels  = [r['l2_rel']  for r in records]
    cos_sims = [r['cos_sim'] for r in records]
    print(f'Global relative L2 drift ({label_a} → {label_b}):')
    print(f'  mean = {np.mean(l2_rels):.4f}')
    print(f'  max  = {np.max(l2_rels):.4f}  ({records[np.argmax(l2_rels)]["name"]})')
    print(f'  min  = {np.min(l2_rels):.4f}')
    print(f'Global cosine similarity:')
    print(f'  mean = {np.mean(cos_sims):.6f}')
    print(f'  min  = {np.min(cos_sims):.6f}  ({records[np.argmin(cos_sims)]["name"]})')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / 'drift_stats.csv'
    fieldnames = ['name', 'group', 'n_params',
                  'l2_abs', 'l2_rel', 'cos_sim',
                  'diff_mean', 'diff_std', 'diff_max']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f'\n  Drift stats CSV: {csv_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating plots …')

    # 1. Relative L2 dot plot
    plot_dot(
        records, key='l2_rel',
        xlabel='Relative L2 drift  ||W₂−W₁||₂ / ||W₁||₂',
        title=f'Per-layer parameter drift: {label_a} → {label_b}',
        out_path=out_dir / 'fig_drift_l2.png',
    )

    # 2. Cosine similarity dot plot
    plot_dot(
        records, key='cos_sim',
        xlabel=f'Cosine similarity ({label_a} vs {label_b} weights)',
        title=f'Per-layer weight cosine similarity: {label_a} vs {label_b}',
        out_path=out_dir / 'fig_drift_cos.png',
    )

    # 3. Weight distribution histograms for top-drift layers
    plot_hist(
        records, sd1, sd2,
        out_path=out_dir / 'fig_drift_hist.png',
        n_layers=args.top_hist,
        label_a=label_a,
        label_b=label_b,
    )

    # 4. Group-level summary (box + dot)
    plot_group_summary(
        records,
        out_path=out_dir / 'fig_drift_groups.png',
        label_a=label_a,
        label_b=label_b,
    )

    print(f'\nAll outputs in: {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
