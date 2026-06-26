#!/usr/bin/env python3
"""
plot_training_analysis.py
-------------------------
Parse Stage 2 results.csv and produce publication-ready Tufte-style figures
for the thesis results section.

results.csv columns (written by train_stage2_ddp.py):
  epoch, loss_box, loss_obj, loss_cls, loss_kd,
  loss_total, mAP50_T2, mAP50_T1, lr

Derived columns computed here:
  loss_det = loss_box + loss_obj + loss_cls
  kd_ratio = loss_kd / loss_det
  FM       = mAP50_T1 − T1_baseline   (requires --t1-baseline, default 0.6617)

Outputs (all PNG, 300 dpi):
  fig_stage2_progress_v2.png  — 4-panel: T2 mAP / T1 mAP / loss components / FM
  fig_loss_decomp_v2.png      — 2-panel: absolute losses / kd:det ratio
  fig_lr_curve.png            — learning rate schedule
  analysis_summary.txt        — key numbers for thesis (peak epoch, FM, etc.)

Usage:
  python plot_training_analysis.py \
      --csv  /projects/prjs2041/runs/stage2/stage2_uda1/results.csv \
      --out-dir /projects/prjs2041/runs/stage2/stage2_uda1/analysis \
      --t1-baseline 0.6617
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d   

# ── Tufte palette ──────────────────────────────────────────────────────────────
BLUE      = '#4878A8'
RED       = '#C94040'
GREEN     = '#5A9B58'
ORANGE    = '#D4823A'
GREY      = '#7F7F7F'
LGREY     = '#E8E8E8'

DPI       = 300


def tufte_ax(ax, grid_axis: str = 'y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GREY)
    ax.spines['bottom'].set_color(GREY)
    ax.tick_params(axis='both', color=GREY, labelsize=8)
    if grid_axis in ('y', 'both'):
        ax.yaxis.grid(True, color=LGREY, linewidth=0.6, zorder=0)
    if grid_axis in ('x', 'both'):
        ax.xaxis.grid(True, color=LGREY, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def vline(ax, x, label=None, color=GREY, lw=1.0, ls='--'):
    ax.axvline(x, color=color, lw=lw, ls=ls, zorder=2)
    if label:
        ax.text(x + 0.3, ax.get_ylim()[1] * 0.97, label,
                fontsize=7, color=color, va='top')


def rolling_avg(series, w: int = 3):
    return uniform_filter1d(series.values.astype(float), size=w, mode='nearest')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — 4-panel progress
# ══════════════════════════════════════════════════════════════════════════════

def plot_stage2_progress(df: pd.DataFrame, t1_baseline: float,
                         out_path: Path):
    """
    Four-panel figure:
      [A] T2 mAP@0.5 with rolling average
      [B] T1 mAP@0.5 with forgetting ceiling
      [C] Loss components (L_det blue, L_kd red)
      [D] Forgetting Measure (FM) with zero reference
    """
    epochs    = df['epoch'].values
    t2_map    = df['mAP50_T2'].values
    t1_map    = df['mAP50_T1'].values
    loss_det  = df['loss_det'].values
    loss_kd   = df['loss_kd'].values
    fm        = df['FM'].values

    best_t2_epoch = int(df.loc[df['mAP50_T2'].idxmax(), 'epoch'])
    best_t2_val   = df['mAP50_T2'].max()
    min_fm_epoch  = int(df.loc[df['FM'].idxmin(), 'epoch'])
    min_fm_val    = df['FM'].min()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=DPI)
    (ax_t2, ax_t1, ax_loss, ax_fm) = axes.flatten()

    for ax in axes.flatten():
        tufte_ax(ax)

    # [A] T2 mAP
    ra = rolling_avg(df['mAP50_T2'], w=3)
    ax_t2.plot(epochs, t2_map, color=BLUE, lw=0.8, alpha=0.35, zorder=2)
    ax_t2.plot(epochs, ra, color=BLUE, lw=1.8, zorder=3, label='3-epoch avg')
    ax_t2.scatter([best_t2_epoch], [best_t2_val], s=55, color=BLUE,
                  zorder=5, marker='*')
    ax_t2.text(best_t2_epoch + 0.5, best_t2_val,
               f'{best_t2_val:.4f} (ep {best_t2_epoch})',
               fontsize=7, color=BLUE, va='center')
    vline(ax_t2, best_t2_epoch, color=BLUE, lw=0.8)
    ax_t2.set_xlabel('Epoch', fontsize=8)
    ax_t2.set_ylabel('mAP@0.5', fontsize=8)
    ax_t2.set_title('(A)  T2 detection performance (Anti-UAV410)',
                    fontsize=9, loc='left', pad=4)

    # [B] T1 mAP with baseline ceiling
    ax_t1.axhline(t1_baseline, color=GREEN, lw=1.0, ls='--', alpha=0.8,
                  label=f'T1 ceiling ({t1_baseline:.4f})')
    ax_t1.fill_between(epochs, t1_map, t1_baseline,
                        where=np.array(t1_map) < t1_baseline,
                        color=RED, alpha=0.10, zorder=1)
    ra_t1 = rolling_avg(df['mAP50_T1'], w=3)
    ax_t1.plot(epochs, t1_map, color=RED, lw=0.8, alpha=0.35, zorder=2)
    ax_t1.plot(epochs, ra_t1, color=RED, lw=1.8, zorder=3)
    vline(ax_t1, best_t2_epoch, color=BLUE, lw=0.8)
    ax_t1.legend(fontsize=7, frameon=False, loc='lower right')
    ax_t1.set_xlabel('Epoch', fontsize=8)
    ax_t1.set_ylabel('mAP@0.5', fontsize=8)
    ax_t1.set_title('(B)  T1 retention (Anti-UAV-RGBT)',
                    fontsize=9, loc='left', pad=4)

    # [C] Loss components
    ax_loss.plot(epochs, loss_det, color=BLUE, lw=1.6, label=r'$L_\mathrm{det}$',
                 zorder=3)
    ax_loss.plot(epochs, loss_kd,  color=RED,  lw=1.6, label=r'$L_\mathrm{kd}$',
                 zorder=3)
    vline(ax_loss, best_t2_epoch, color=BLUE, lw=0.8)
    ax_loss.legend(fontsize=8, frameon=False)
    ax_loss.set_xlabel('Epoch', fontsize=8)
    ax_loss.set_ylabel('Loss', fontsize=8)
    ax_loss.set_title(r'(C)  Loss components ($L_\mathrm{det}$ vs $L_\mathrm{kd}$)',
                      fontsize=9, loc='left', pad=4)

    # [D] Forgetting Measure
    ax_fm.axhline(0, color=GREY, lw=0.8, ls='-', alpha=0.6)
    ax_fm.axhline(-0.05, color=RED, lw=0.8, ls='--', alpha=0.7,
                  label='−0.05 threshold')
    ax_fm.fill_between(epochs, fm, 0,
                        where=np.array(fm) < 0,
                        color=RED, alpha=0.12, zorder=1)
    ax_fm.plot(epochs, fm, color=ORANGE, lw=1.6, zorder=3)
    ax_fm.scatter([min_fm_epoch], [min_fm_val], s=55, color=RED,
                  zorder=5, marker='v')
    ax_fm.text(min_fm_epoch + 0.4, min_fm_val,
               f'min FM={min_fm_val:.4f}\n(ep {min_fm_epoch})',
               fontsize=7, color=RED, va='top')
    vline(ax_fm, best_t2_epoch, color=BLUE, lw=0.8)
    ax_fm.legend(fontsize=7, frameon=False, loc='lower right')
    ax_fm.set_xlabel('Epoch', fontsize=8)
    ax_fm.set_ylabel('FM', fontsize=8)
    ax_fm.set_title('(D)  Forgetting Measure (FM)', fontsize=9, loc='left', pad=4)

    # Common x-limit
    for ax in axes.flatten():
        ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

    # Blue vertical reference annotation (shared across panels)
    fig.text(0.5, 0.01,
             f'Blue dashed line: peak T2 mAP epoch ({best_t2_epoch})',
             ha='center', fontsize=7, color=BLUE, alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Loss decomposition 2-panel
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_decomp(df: pd.DataFrame, out_path: Path):
    """
    Two-panel:
      Left  — absolute L_det and L_kd per epoch
      Right — kd/det ratio with vertical line at peak T2 epoch
    """
    epochs    = df['epoch'].values
    loss_det  = df['loss_det'].values
    loss_kd   = df['loss_kd'].values
    kd_ratio  = df['kd_ratio'].values

    best_t2_epoch = int(df.loc[df['mAP50_T2'].idxmax(), 'epoch'])

    fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(10, 4), dpi=DPI)

    for ax in (ax_abs, ax_ratio):
        tufte_ax(ax)

    # Left — absolute losses
    ax_abs.plot(epochs, loss_det, color=BLUE, lw=2.0,
                label=r'$L_\mathrm{det}$', zorder=3)
    ax_abs.plot(epochs, loss_kd,  color=RED,  lw=2.0,
                label=r'$L_\mathrm{kd}$',  zorder=3)

    # Annotate endpoints
    for vals, col, name in [
        (loss_det, BLUE, r'$L_\mathrm{det}$'),
        (loss_kd,  RED,  r'$L_\mathrm{kd}$'),
    ]:
        ax_abs.text(epochs[-1] + 0.3, vals[-1], name,
                    color=col, fontsize=8, va='center')

    ax_abs.axvline(best_t2_epoch, color=BLUE, lw=1.0, ls='--', alpha=0.7)
    ax_abs.set_xlabel('Epoch', fontsize=9)
    ax_abs.set_ylabel('Loss value', fontsize=9)
    ax_abs.set_title('Absolute loss components per epoch',
                     fontsize=10, loc='left')
    ax_abs.set_xlim(epochs[0] - 0.5, epochs[-1] + 2)

    # Right — kd/det ratio
    ax_ratio.plot(epochs, kd_ratio, color=ORANGE, lw=2.0, zorder=3)
    ax_ratio.axvline(best_t2_epoch, color=BLUE, lw=1.0, ls='--', alpha=0.7)
    ax_ratio.text(best_t2_epoch + 0.3,
                  ax_ratio.get_ylim()[1] if ax_ratio.get_ylim()[1] != 1.0 else kd_ratio.max() * 0.97,
                  f'ep {best_t2_epoch}',
                  fontsize=7, color=BLUE, va='top')

    # Annotate start/end ratio values
    ax_ratio.annotate(
        f'{kd_ratio[0]:.2f}×',
        xy=(epochs[0], kd_ratio[0]),
        xytext=(epochs[0] + 1, kd_ratio[0] + 0.05),
        fontsize=7, color=ORANGE,
        arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8),
    )
    ax_ratio.annotate(
        f'{kd_ratio[-1]:.2f}×',
        xy=(epochs[-1], kd_ratio[-1]),
        xytext=(epochs[-1] - 3, kd_ratio[-1] + 0.05),
        fontsize=7, color=ORANGE,
        arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8),
    )

    ax_ratio.set_xlabel('Epoch', fontsize=9)
    ax_ratio.set_ylabel(r'$L_\mathrm{kd}$ / $L_\mathrm{det}$ ratio', fontsize=9)
    ax_ratio.set_title(r'KD-to-detection loss ratio',
                       fontsize=10, loc='left')
    ax_ratio.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Learning rate schedule
# ══════════════════════════════════════════════════════════════════════════════

def plot_lr(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=DPI)
    tufte_ax(ax)

    epochs = df['epoch'].values
    lr     = df['lr'].values

    ax.plot(epochs, lr, color=GREEN, lw=1.8, zorder=3)
    ax.fill_between(epochs, lr, alpha=0.12, color=GREEN, zorder=1)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Learning rate', fontsize=9)
    ax.set_title('Cosine learning rate schedule (Stage 2)',
                 fontsize=10, loc='left')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, -3))
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Text summary
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(df: pd.DataFrame, t1_baseline: float, out_path: Path):
    best_t2_row = df.loc[df['mAP50_T2'].idxmax()]
    min_fm_row  = df.loc[df['FM'].idxmin()]
    last_row    = df.iloc[-1]
    first_row   = df.iloc[0]

    lines = [
        '=' * 60,
        'Stage 2 Training Analysis — Key Numbers',
        '=' * 60,
        '',
        f'Total epochs trained       : {int(last_row["epoch"]) + 1}',
        f'Early stopping triggered at: epoch {int(last_row["epoch"])}',
        '',
        '--- T2 performance (Anti-UAV410 val) ---',
        f'Peak T2 mAP@0.5            : {best_t2_row["mAP50_T2"]:.4f}  (epoch {int(best_t2_row["epoch"])})',
        f'Final T2 mAP@0.5           : {last_row["mAP50_T2"]:.4f}',
        '',
        '--- T1 retention (Anti-UAV-RGBT val) ---',
        f'T1 baseline (Stage 1)      : {t1_baseline:.4f}',
        f'Best T1 mAP during T2      : {df["mAP50_T1"].max():.4f}  (epoch {int(df.loc[df["mAP50_T1"].idxmax(), "epoch"])})',
        f'FM (best T1 − baseline)    : {df["mAP50_T1"].max() - t1_baseline:.4f}',
        '',
        '--- Forgetting Measure (FM = mAP_T1_after_T2 − T1_baseline) ---',
        f'Min FM (max forgetting)    : {min_fm_row["FM"]:.4f}  (epoch {int(min_fm_row["epoch"])})',
        f'FM at peak T2 epoch        : {best_t2_row["FM"]:.4f}',
        f'Final FM                   : {last_row["FM"]:.4f}',
        '',
        '--- Loss decomposition ---',
        f'L_det: epoch 0  = {first_row["loss_det"]:.4f}   →  epoch {int(last_row["epoch"])} = {last_row["loss_det"]:.4f}',
        f'L_kd:  epoch 0  = {first_row["loss_kd"]:.4f}   →  epoch {int(last_row["epoch"])} = {last_row["loss_kd"]:.4f}',
        f'kd/det ratio: epoch 0 = {first_row["kd_ratio"]:.2f}×  →  epoch {int(last_row["epoch"])} = {last_row["kd_ratio"]:.2f}×',
        f'(at peak T2 epoch {int(best_t2_row["epoch"])}: kd/det = {best_t2_row["kd_ratio"]:.2f}×)',
        '',
        '--- Stability-plasticity gap ---',
        f'Peak T2 epoch              : {int(best_t2_row["epoch"])}',
        f'Min FM epoch               : {int(min_fm_row["epoch"])}',
        f'Gap                        : {abs(int(best_t2_row["epoch"]) - int(min_fm_row["epoch"]))} epochs',
        '',
        '=' * 60,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    out_path.write_text(text)
    print(f'\n  Summary: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def plot_stage3_progress(df: pd.DataFrame, t1_baseline: float, out_path: Path):
    """
    Four-panel figure for Stage 3:
      [A] T3 mAP@0.5 (CST val) — new task performance
      [B] T1 mAP@0.5 (Anti-UAV-RGBT val) — forgetting curve
      [C] Per-stratum T1 mAP@0.5 (tiny / small / normal / large) — KEY FIGURE
      [D] Forgetting Measure (fm_abs and fm_stage3)
    """
    epochs = df['epoch'].values

    best_t3_epoch = int(df.loc[df['mAP50_T3'].idxmax(), 'epoch'])
    best_t3_val   = df['mAP50_T3'].max()
    max_fm_epoch  = int(df.loc[df['fm_abs'].idxmin(), 'epoch'])
    max_fm_val    = df['fm_abs'].min()

    STRATUM_COLORS = {
        'tiny':   '#d62728',   # red
        'small':  '#ff7f0e',   # orange
        'normal': '#2ca02c',   # green
        'large':  '#1f77b4',   # blue
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=DPI)
    (ax_t3, ax_t1, ax_strat, ax_fm) = axes.flatten()
    for ax in axes.flatten():
        tufte_ax(ax)

    # [A] T3 mAP
    ra_t3 = rolling_avg(df['mAP50_T3'], w=3)
    ax_t3.plot(epochs, df['mAP50_T3'].values, color=BLUE, lw=0.8, alpha=0.35, zorder=2)
    ax_t3.plot(epochs, ra_t3, color=BLUE, lw=1.8, zorder=3, label='3-epoch avg')
    ax_t3.scatter([best_t3_epoch], [best_t3_val], s=55, color=BLUE, zorder=5, marker='*')
    ax_t3.text(best_t3_epoch + 0.3, best_t3_val,
               f'{best_t3_val:.4f} (ep {best_t3_epoch})',
               fontsize=7, color=BLUE, va='center')
    vline(ax_t3, best_t3_epoch, color=BLUE, lw=0.8)
    ax_t3.set_xlabel('Epoch', fontsize=8)
    ax_t3.set_ylabel('mAP@0.5', fontsize=8)
    ax_t3.set_title('(A)  T3 detection performance (CST Anti-UAV)',
                    fontsize=9, loc='left', pad=4)

    # [B] T1 mAP
    ax_t1.axhline(t1_baseline, color=GREEN, lw=1.0, ls='--', alpha=0.8,
                  label=f'T1 ceiling ({t1_baseline:.4f})')
    t1_vals = df['mAP50_T1'].values
    ax_t1.fill_between(epochs, t1_vals, t1_baseline,
                       where=t1_vals < t1_baseline,
                       color=RED, alpha=0.10, zorder=1)
    ra_t1 = rolling_avg(df['mAP50_T1'], w=3)
    ax_t1.plot(epochs, t1_vals, color=RED, lw=0.8, alpha=0.35, zorder=2)
    ax_t1.plot(epochs, ra_t1, color=RED, lw=1.8, zorder=3)
    vline(ax_t1, best_t3_epoch, color=BLUE, lw=0.8)
    ax_t1.legend(fontsize=7, frameon=False, loc='upper right')
    ax_t1.set_xlabel('Epoch', fontsize=8)
    ax_t1.set_ylabel('mAP@0.5', fontsize=8)
    ax_t1.set_title('(B)  T1 retention (Anti-UAV-RGBT val)',
                    fontsize=9, loc='left', pad=4)

    # [C] Per-stratum T1 mAP — the key figure
    for stratum, color in STRATUM_COLORS.items():
        col = f'mAP50_T1_{stratum}'
        if col in df.columns:
            vals = df[col].values
            ax_strat.plot(epochs, vals, color=color, lw=1.8, zorder=3,
                          label=stratum.capitalize())
            # Annotate epoch 2 value (large collapse point)
            if stratum == 'large' and len(epochs) > 2:
                ax_strat.annotate(
                    f'{vals[2]:.3f}',
                    xy=(epochs[2], vals[2]),
                    xytext=(epochs[2] + 0.5, vals[2] + 0.03),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.7),
                )
    vline(ax_strat, best_t3_epoch, color=BLUE, lw=0.8)
    ax_strat.legend(fontsize=8, frameon=False, loc='upper right',
                    title='Stratum', title_fontsize=7)
    ax_strat.set_xlabel('Epoch', fontsize=8)
    ax_strat.set_ylabel('mAP@0.5', fontsize=8)
    ax_strat.set_title('(C)  Per-stratum T1 mAP@0.5  [scale-specific erasure]',
                       fontsize=9, loc='left', pad=4)

    # [D] Forgetting Measure
    ax_fm.axhline(0, color=GREY, lw=0.8, ls='-', alpha=0.6)
    ax_fm.plot(epochs, df['fm_abs'].values,    color=RED,    lw=1.8,
               zorder=3, label=r'$FM_{abs}$ (vs T1 ceiling)')
    ax_fm.plot(epochs, df['fm_stage3'].values, color=ORANGE, lw=1.8,
               ls='--', zorder=3, label=r'$FM_{stage3}$ (vs Stage 2 baseline)')
    ax_fm.scatter([max_fm_epoch], [max_fm_val], s=55, color=RED, zorder=5, marker='v')
    ax_fm.text(max_fm_epoch + 0.3, max_fm_val,
               f'min={max_fm_val:.4f}\n(ep {max_fm_epoch})',
               fontsize=7, color=RED, va='top')
    vline(ax_fm, best_t3_epoch, color=BLUE, lw=0.8)
    ax_fm.legend(fontsize=8, frameon=False, loc='upper right', bbox_to_anchor=(1.02, 0.95))
    ax_fm.set_xlabel('Epoch', fontsize=8)
    ax_fm.set_ylabel('FM', fontsize=8)
    ax_fm.set_title('(D)  Forgetting Measure (Stage 3)',
                    fontsize=9, loc='left', pad=4)

    for ax in axes.flatten():
        ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

    fig.text(0.5, 0.01,
             f'Blue dashed line: best T3 mAP epoch ({best_t3_epoch})',
             ha='center', fontsize=7, color=BLUE, alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def write_stage3_summary(df: pd.DataFrame, t1_baseline: float, out_path: Path):
    best_t3_row = df.loc[df['mAP50_T3'].idxmax()]
    min_fm_row  = df.loc[df['fm_abs'].idxmin()]
    last_row    = df.iloc[-1]
    first_row   = df.iloc[0]

    lines = [
        '=' * 60,
        'Stage 3 Training Analysis — Key Numbers',
        '=' * 60,
        '',
        f'Total epochs trained        : {int(last_row["epoch"]) + 1}',
        f'Early stopping triggered at : epoch {int(last_row["epoch"])}',
        '',
        '--- T3 performance (CST Anti-UAV val) ---',
        f'Peak T3 mAP@0.5             : {best_t3_row["mAP50_T3"]:.4f}  (epoch {int(best_t3_row["epoch"])})',
        f'Final T3 mAP@0.5            : {last_row["mAP50_T3"]:.4f}',
        '',
        '--- T1 retention (Anti-UAV-RGBT val) ---',
        f'T1 ceiling (Stage 1)        : {t1_baseline:.4f}',
        f'T1 mAP at best T3 epoch     : {best_t3_row["mAP50_T1"]:.4f}',
        f'T1 mAP at final epoch       : {last_row["mAP50_T1"]:.4f}',
        '',
        '--- Forgetting Measure at best T3 epoch ---',
        f'fm_abs   (vs T1 ceiling)    : {best_t3_row["fm_abs"]:.4f}',
        f'fm_stage3 (vs Stage 2 ret.) : {best_t3_row["fm_stage3"]:.4f}',
        f'Worst fm_abs (min across epochs): {min_fm_row["fm_abs"]:.4f}  (epoch {int(min_fm_row["epoch"])})',
        '',
        '--- Per-stratum T1 mAP at best T3 epoch ---',
        *[f'{s:<8}: {best_t3_row.get(f"mAP50_T1_{s}", float("nan")):.4f}'
          for s in ['tiny', 'small', 'normal', 'large']],
        '',
        '--- Loss at final epoch ---',
        f'loss_cst    : {last_row["loss_cst"]:.4f}',
        f'loss_replay : {last_row["loss_replay"]:.4f}',
        '',
        '=' * 60,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    out_path.write_text(text)
    print(f'\n  Summary: {out_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=None,
                   help='Path to results.csv (Stage 1/2/3; auto-discovered from cwd if omitted)')
    p.add_argument('--out-dir', type=str, default=None,
                   help='Output directory (default: same dir as results.csv)')
    p.add_argument('--t1-baseline', type=float, default=0.6725,
                   help='Stage 1 T1 mAP@0.5 (for FM computation). '
                        'Updated to 0.6725 from antiuav_rgbt15.')
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-discover results.csv
    if args.csv:
        csv_path = Path(args.csv)
    else:
        candidates = list(Path('.').glob('results.csv'))
        if not candidates:
            print('[ERROR] No results.csv found. Use --csv to specify path.')
            sys.exit(1)
        csv_path = candidates[0]

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading: {csv_path}')
    df = pd.read_csv(csv_path)

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # ── Stage detection ────────────────────────────────────────────────────────
    # Stage 3: epoch, loss_cst, loss_replay, loss_total,
    #          mAP50_T3, P_T3, R_T3, F1_T3,
    #          mAP50_T1, P_T1, R_T1, F1_T1,
    #          fm_abs, fm_stage3,
    #          mAP50_T1_tiny, mAP50_T1_small, mAP50_T1_normal, mAP50_T1_large, lr
    # Stage 2: epoch, loss_box, loss_obj, loss_cls, loss_kd,
    #          loss_total, mAP50_T2, P_T2, R_T2, F1_T2,
    #          mAP50_T1, P_T1, R_T1, F1_T1, lr
    # Stage 1: epoch, loss_box, loss_obj, loss_cls, loss_total,
    #          mAP50, mAP50-95, precision, recall, f1, lr
    is_stage3 = 'loss_cst' in df.columns and 'fm_abs' in df.columns
    is_stage2 = (not is_stage3) and ('loss_kd' in df.columns)

    # ── Route to the right pipeline ───────────────────────────────────────────
    if is_stage3:
        stage = 'Stage 3'
        print(f'  {len(df)} epochs loaded  [{stage} format]  '
              f'(cols: {list(df.columns)})\n')

        plot_stage3_progress(
            df, args.t1_baseline,
            out_dir / 'fig_stage3_progress.png'
        )
        plot_lr(df, out_dir / 'fig_lr_curve.png')
        write_stage3_summary(df, args.t1_baseline, out_dir / 'analysis_summary.txt')

        # Save a copy of the CSV for reference
        df.to_csv(out_dir / 'results_extended.csv', index=False)
        print(f'\n  Extended CSV: {out_dir / "results_extended.csv"}')

    else:
        # ── Stage 1 / Stage 2 pipeline (requires loss_box/obj/cls columns) ──
        df['loss_det'] = df['loss_box'] + df['loss_obj'] + df['loss_cls']

        if is_stage2:
            df['kd_ratio'] = df['loss_kd'] / (df['loss_det'] + 1e-12)
            df['FM']       = df['mAP50_T1'] - args.t1_baseline
            stage = 'Stage 2'
        else:
            # Stage 1: stub out Stage-2 columns so plot functions work
            df['loss_kd']  = 0.0
            df['mAP50_T2'] = df['mAP50']
            df['mAP50_T1'] = args.t1_baseline
            df['kd_ratio'] = 0.0
            df['FM']       = 0.0
            if 'precision' in df.columns:
                df['P_T2']  = df['precision']
                df['R_T2']  = df['recall']
                df['F1_T2'] = df['f1']
            stage = 'Stage 1'

        print(f'  {len(df)} epochs loaded  [{stage} format]  '
              f'(cols: {list(df.columns)})\n')

        plot_stage2_progress(
            df, args.t1_baseline,
            out_dir / f'fig_{stage.lower().replace(" ", "")}_progress.png'
        )
        plot_loss_decomp(df, out_dir / 'fig_loss_decomp_v2.png')
        plot_lr(df, out_dir / 'fig_lr_curve.png')
        write_summary(df, args.t1_baseline, out_dir / 'analysis_summary.txt')

        extended_csv = out_dir / 'results_extended.csv'
        df.to_csv(extended_csv, index=False)
        print(f'\n  Extended CSV: {extended_csv}')

    print(f'\nAll outputs in: {out_dir}')
    print('Done.')


if __name__ == '__main__':
    main()
