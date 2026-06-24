#!/usr/bin/env python3
"""
eval_stratum_t1.py
------------------
Per-stratum T1 (Anti-UAV-RGBT) mAP@0.5 for one or more checkpoints, on the
SAME split used for the Forgetting Measure. Produces the column missing from
Table 1: per-stratum T1 retention AFTER STAGE 2, alongside After Stage 1 and
After Stage 3 for direct comparison.

WHY
---
Overall T1 mAP stays high after Stage 2 (95% retention), but that aggregate
could be genuine preservation OR a flattering overlap: Anti-UAV410 (Stage 2)
shares T1's large-to-normal scale regime, so it may broaden the same
representation rather than preserve it. A per-stratum breakdown after Stage 2
settles which. If the large stratum still sits near its Stage-1 value (0.461),
KD truly preserved it; if it has already slipped, the 95% figure is carried by
the strata that T2 happens to overlap.

This is EVAL ONLY -- no training, no optimiser. Minutes on one GPU.

It reuses the exact evaluation code from train_stage3.py (same NMS thresholds,
same stratum bins), so the S1 and S3 columns should reproduce Table 1. Use that
as a sanity check before trusting the new S2 column.

USAGE
-----
  python eval_stratum_t1.py \
      --s1 /projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt \
      --s2 /projects/prjs2041/runs/stage2/seed42/weights/best.pt \
      --s3 /projects/prjs2041/runs/stage3/naive2/weights/best.pt \
      --split test \
      --out /projects/prjs2041/runs/diagnostics/stratum_t1

Pass any subset of --s1/--s2/--s3. --split should match whatever Table 1 used
(Table 1 was computed on the Anti-UAV-RGBT validation split).
"""

import csv
import argparse
from pathlib import Path

import torch

# Reuse model construction, datasets, and the exact per-stratum eval from Stage 3.
import train_stage3 as s3


class RGBTSplitDataset(torch.utils.data.Dataset):
    """Anti-UAV-RGBT wrapper for an arbitrary split (img2 = zeros)."""

    def __init__(self, root, split='test', imgsz=640):
        self.ds = s3.AntiUAVRGBTDataset(root=root, split=split)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = s3._letterbox_tensor(sample['image'], self.imgsz)
        img2 = torch.zeros_like(img)
        return img, img2, sample['labels'], sample.get('meta', {})


def load_model(weights, device):
    """Load a checkpoint into YOLOMG exactly as train_stage3 does (EMA preferred)."""
    ckpt = torch.load(str(weights), map_location=device, weights_only=False)
    model = s3.Model(s3.MODEL_CFG, ch=3, ch2=3, nc=s3.NC).to(device)
    model.nc = s3.NC
    model.hyp = s3.HYP
    model.names = s3.NAMES
    state = ckpt.get('ema') or ckpt.get('model')
    sd = state.float().state_dict() if state else {}
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'    ({len(missing)} missing keys)')
    model.eval()
    return model


def eval_checkpoint(weights, loader, device):
    model = load_model(Path(weights), device)
    stats, sizes = s3._collect_stats_with_sizes(model, loader, device, s3.IMG_SIZE)
    strata = s3._stratum_map(stats, sizes)          # dict: stratum -> mAP50
    overall = s3._compute_metrics(stats)[0]         # overall mAP50
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return strata, overall


def main():
    ap = argparse.ArgumentParser(description='Per-stratum T1 mAP for S1/S2/S3 checkpoints.')
    ap.add_argument('--s1', type=str, default=None, help='Stage 1 best.pt')
    ap.add_argument('--s2', type=str, default=None, help='Stage 2 best.pt')
    ap.add_argument('--s3', type=str, default=None, help='Stage 3 best.pt')
    ap.add_argument('--split', type=str, default='val',
                    help="Anti-UAV-RGBT split (match Table 1; default 'val').")
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--rgbt-root', type=str, default=str(s3.RGBT_ROOT))
    ap.add_argument('--out', type=str,
                    default='/projects/prjs2041/runs/diagnostics/stratum_t1')
    args = ap.parse_args()

    cols = []
    if args.s1:
        cols.append(('After Stage 1', args.s1))
    if args.s2:
        cols.append(('After Stage 2', args.s2))
    if args.s3:
        cols.append(('After Stage 3', args.s3))
    if not cols:
        raise SystemExit('Provide at least one of --s1 / --s2 / --s3')

    device = s3.select_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = RGBTSplitDataset(Path(args.rgbt_root), split=args.split, imgsz=s3.IMG_SIZE)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=s3._collate)
    print(f'\nAnti-UAV-RGBT {args.split} split: {len(ds):,} frames\n')

    results = {}
    for label, path in cols:
        print(f'Evaluating {label}: {path}')
        results[label] = eval_checkpoint(path, loader, device)

    strata_names = [c[0] for c in s3.SIZE_BINS_T1]   # tiny, small, normal, large

    print('\n' + '=' * 72)
    print(f'PER-STRATUM T1 mAP@0.5 on Anti-UAV-RGBT {args.split} split')
    print('=' * 72)
    print(f'{"Stratum":<16}' + ''.join(f'{lbl:>18}' for lbl, _ in cols))
    for s in strata_names:
        line = f'{s:<16}'
        for lbl, _ in cols:
            line += f'{results[lbl][0].get(s, float("nan")):>18.3f}'
        print(line)
    line = f'{"Overall":<16}'
    for lbl, _ in cols:
        line += f'{results[lbl][1]:>18.3f}'
    print(line)
    print('=' * 72)

    # If both S1 and S2 present, print the headline check directly.
    if 'After Stage 1' in results and 'After Stage 2' in results:
        l1 = results['After Stage 1'][0].get('large', float('nan'))
        l2 = results['After Stage 2'][0].get('large', float('nan'))
        print(f'\nLarge-stratum T1 mAP:  after S1 = {l1:.3f}  ->  after S2 = {l2:.3f}'
              f'   (drop = {l2 - l1:+.3f})')
        print('If close, KD genuinely preserved the large stratum; if it has already '
              'dropped, the 95% overall figure is partly T1/T2 scale overlap.\n')

    csv_path = out_dir / f'stratum_t1_{args.split}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stratum'] + [lbl for lbl, _ in cols])
        for s in strata_names:
            w.writerow([s] + [f'{results[lbl][0].get(s, float("nan")):.4f}' for lbl, _ in cols])
        w.writerow(['overall'] + [f'{results[lbl][1]:.4f}' for lbl, _ in cols])
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()
