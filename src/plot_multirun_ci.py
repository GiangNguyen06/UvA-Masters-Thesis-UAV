#!/usr/bin/env python3
"""
plot_multirun_ci.py
-------------------
Aggregate results.csv from multiple Stage 2 seed runs and produce
mean ± std confidence-interval plots for the thesis.

Usage:
  python plot_multirun_ci.py \
      --runs-root /projects/prjs2041/runs/stage2 \
      --pattern   "seed*" \
      --out-dir   /projects/prjs2041/runs/stage2/ci_plots \
      --t1-baseline 0.6617

Expects results.csv in each matched subdirectory with columns:
  epoch, loss_box, loss_obj, loss_cls, loss_kd, loss_total,
  mAP50_T2, P_T2, R_T2, F1_T2,
  mAP50_T1, P_T1, R_T1, F1_T1, lr
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE   = '#4878A8'
RED    = '#C94040'
GREEN  = '#5A9B58'
ORANGE = '#D4823A'
GREY   = '#7F7F7F'
LGREY  = '#E8E8E8'
DPI    = 300


def tufte_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GREY)
    ax.spines['bottom'].set_color(GREY)
    ax.tick_params(axis='both', color=GREY, labelsize=8)
    ax.yaxis.grid(True, color=LGREY, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def load_runs(runs_root: Path, pattern: str) -> list[pd.DataFrame]:
    """Load results.csv from all matching subdirectories."""
    dirs = sorted(runs_root.glob(pattern))
    dfs  = []
    for d in dirs:
        csv_path = d / 'results.csv'
        if not csv_path.exists():
            print(f'  [SKIP] {d.name} — no results.csv')
            continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Add derived columns if not already present
        if 'loss_det' not in df.columns:
            df['loss_det'] = df['loss_box'] + df['loss_obj'] + df['loss_cls']
        if 'kd_ratio' not in df.columns:
            df['kd_ratio'] = df['loss_kd'] / (df['loss_det'] + 1e-12)

        df['run'] = d.name
        dfs.append(df)
        print(f'  Loaded: {d.name}  ({len(df)} epochs)')
    return dfs


def align_epochs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Align runs to a common epoch range (intersection), then compute
    mean and std across runs for each metric at each epoch.
    """
    # Find common epochs
    epoch_sets = [set(df['epoch'].values) for df in dfs]
    common = sorted(set.intersection(*epoch_sets))
    print(f'  Common epochs: {common[0]}–{common[-1]}  ({len(common)} epochs)')

    filtered = [df[df['epoch'].isin(common)].set_index('epoch') for df in dfs]
    combined = pd.concat(filtered)

    # Drop non-numeric columns (e.g. 'run' = seed name) before aggregating
    combined = combined.select_dtypes(include='number')

    agg = combined.groupby('epoch').agg(['mean', 'std'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index()
    return agg


def plot_ci_panel(ax, epochs, mean, std, colour, label, alpha_band=0.15):
    ax.fill_between(epochs, mean - std, mean + std,
                    color=colour, alpha=alpha_band, zorder=1)
    ax.plot(epochs, mean, color=colour, lw=1.8, zorder=3, label=label)


def plot_ci_figure(agg: pd.DataFrame, t1_baseline: float, out_path: Path,
                   n_seeds: int):
    """4-panel CI figure."""
    epochs = agg['epoch'].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=DPI)
    (ax_t2, ax_t1, ax_loss, ax_fm) = axes.flatten()
    for ax in axes.flatten():
        tufte_ax(ax)

    # Helper: column exists check
    def col(name, fallback=None):
        return agg[name] if name in agg.columns else \
               (pd.Series(np.zeros(len(agg))) if fallback is None else fallback)

    # [A] T2 mAP
    plot_ci_panel(ax_t2, epochs,
                  col('mAP50_T2_mean').values, col('mAP50_T2_std').values,
                  BLUE, f'mAP@0.5 T2 (n={n_seeds})')
    ax_t2.set_xlabel('Epoch', fontsize=8)
    ax_t2.set_ylabel('mAP@0.5', fontsize=8)
    ax_t2.set_title('(A)  T2 mAP@0.5 — Anti-UAV410 val', fontsize=9, loc='left')
    ax_t2.legend(fontsize=7, frameon=False)

    # [B] T1 mAP with ceiling
    ax_t1.axhline(t1_baseline, color=GREEN, lw=1.0, ls='--', alpha=0.8,
                  label=f'T1 ceiling ({t1_baseline:.4f})')
    plot_ci_panel(ax_t1, epochs,
                  col('mAP50_T1_mean').values, col('mAP50_T1_std').values,
                  RED, f'mAP@0.5 T1 (n={n_seeds})')
    ax_t1.set_xlabel('Epoch', fontsize=8)
    ax_t1.set_ylabel('mAP@0.5', fontsize=8)
    ax_t1.set_title('(B)  T1 mAP@0.5 — Anti-UAV-RGBT val', fontsize=9, loc='left')
    ax_t1.legend(fontsize=7, frameon=False)

    # [C] F1 for both datasets
    if 'F1_T2_mean' in agg.columns:
        plot_ci_panel(ax_loss, epochs,
                      col('F1_T2_mean').values, col('F1_T2_std').values,
                      BLUE, 'F1 T2')
        plot_ci_panel(ax_loss, epochs,
                      col('F1_T1_mean').values, col('F1_T1_std').values,
                      RED, 'F1 T1')
        ax_loss.set_ylabel('F1', fontsize=8)
        ax_loss.set_title('(C)  F1 score (T1 and T2)', fontsize=9, loc='left')
    else:
        # Fallback: loss components
        plot_ci_panel(ax_loss, epochs,
                      col('loss_det_mean').values, col('loss_det_std').values,
                      BLUE, r'$L_\mathrm{det}$')
        plot_ci_panel(ax_loss, epochs,
                      col('loss_kd_mean').values, col('loss_kd_std').values,
                      RED, r'$L_\mathrm{kd}$')
        ax_loss.set_ylabel('Loss', fontsize=8)
        ax_loss.set_title('(C)  Loss components', fontsize=9, loc='left')
    ax_loss.set_xlabel('Epoch', fontsize=8)
    ax_loss.legend(fontsize=7, frameon=False)

    # [D] Forgetting Measure
    fm_mean = col('mAP50_T1_mean').values - t1_baseline
    fm_std  = col('mAP50_T1_std').values
    ax_fm.axhline(0, color=GREY, lw=0.8, ls='-', alpha=0.5)
    ax_fm.axhline(-0.05, color=RED, lw=0.8, ls='--', alpha=0.6,
                  label='−0.05 threshold')
    plot_ci_panel(ax_fm, epochs, fm_mean, fm_std, ORANGE,
                  f'FM (n={n_seeds})')
    ax_fm.set_xlabel('Epoch', fontsize=8)
    ax_fm.set_ylabel('FM', fontsize=8)
    ax_fm.set_title('(D)  Forgetting Measure (FM)', fontsize=9, loc='left')
    ax_fm.legend(fontsize=7, frameon=False)

    for ax in axes.flatten():
        ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def write_ci_summary(agg: pd.DataFrame, t1_baseline: float,
                     dfs: list[pd.DataFrame], out_path: Path):
    """Write a text summary of key statistics across seeds."""
    lines = ['=' * 65,
             f'Multi-seed Confidence Interval Summary  (n={len(dfs)} seeds)',
             '=' * 65, '']

    # Best T2 mAP per seed
    lines.append('--- Peak T2 mAP@0.5 per seed ---')
    peak_t2s = []
    for df in dfs:
        peak = df['mAP50_T2'].max()
        ep   = int(df.loc[df['mAP50_T2'].idxmax(), 'epoch'])
        lines.append(f'  {df["run"].iloc[0]:<20}  peak={peak:.4f}  epoch={ep}')
        peak_t2s.append(peak)
    lines.append(f'  MEAN ± STD : {np.mean(peak_t2s):.4f} ± {np.std(peak_t2s):.4f}')
    lines.append('')

    # FM at peak T2 epoch per seed
    lines.append('--- FM at peak T2 epoch per seed ---')
    fms = []
    for df in dfs:
        ep  = int(df.loc[df['mAP50_T2'].idxmax(), 'epoch'])
        fm  = float(df[df['epoch'] == ep]['mAP50_T1'].values[0]) - t1_baseline
        lines.append(f'  {df["run"].iloc[0]:<20}  FM={fm:.4f}  (epoch {ep})')
        fms.append(fm)
    lines.append(f'  MEAN ± STD : {np.mean(fms):.4f} ± {np.std(fms):.4f}')
    lines.append('')

    # F1 at peak T2 epoch (if available)
    if 'F1_T2' in dfs[0].columns:
        lines.append('--- F1_T2 at peak T2 epoch per seed ---')
        f1s = []
        for df in dfs:
            ep = int(df.loc[df['mAP50_T2'].idxmax(), 'epoch'])
            f1 = float(df[df['epoch'] == ep]['F1_T2'].values[0])
            lines.append(f'  {df["run"].iloc[0]:<20}  F1={f1:.4f}')
            f1s.append(f1)
        lines.append(f'  MEAN ± STD : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
        lines.append('')

    lines += ['=' * 65]
    text = '\n'.join(lines)
    print('\n' + text)
    out_path.write_text(text)
    print(f'\n  Summary: {out_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-root',    type=str,
                   default='/projects/prjs2041/runs/stage2')
    p.add_argument('--pattern',      type=str, default='seed*')
    p.add_argument('--out-dir',      type=str, default=None)
    p.add_argument('--t1-baseline',  type=float, default=0.6617)
    return p.parse_args()


def main():
    args     = parse_args()
    runs_root = Path(args.runs_root)
    out_dir   = Path(args.out_dir) if args.out_dir \
                else runs_root / 'ci_plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading runs matching: {runs_root}/{args.pattern}')
    dfs = load_runs(runs_root, args.pattern)
    if len(dfs) < 2:
        print('[ERROR] Need at least 2 runs for CI. Found:', len(dfs))
        return

    print(f'\nAligning epochs across {len(dfs)} seeds ...')
    agg = align_epochs(dfs)

    print('\nGenerating plots ...')
    plot_ci_figure(agg, args.t1_baseline,
                   out_dir / 'fig_stage2_ci.png',
                   n_seeds=len(dfs))

    write_ci_summary(agg, args.t1_baseline, dfs,
                     out_dir / 'ci_summary.txt')

    # Save aggregated CSV
    agg_csv = out_dir / 'agg_results.csv'
    agg.to_csv(agg_csv, index=False)
    print(f'  Aggregated CSV: {agg_csv}')

    print(f'\nAll outputs in: {out_dir}')


if __name__ == '__main__':
    main()
