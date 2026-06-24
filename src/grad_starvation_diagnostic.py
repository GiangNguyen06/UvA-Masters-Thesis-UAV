#!/usr/bin/env python3
"""
grad_starvation_diagnostic.py
-----------------------------
Direct measurement of gradient starvation at the Stage 2 -> Stage 3 boundary.

WHY THIS EXISTS
---------------
The thesis argues that large-target forgetting in Stage 3 is driven by
*gradient starvation*: because CST Anti-UAV contains 0% large targets, the
large-target detection pathway receives essentially no gradient signal during
Stage 3 fine-tuning, so those features decay without being actively overwritten.
So far that claim is INFERRED from the cosine-similarity / BatchNorm evidence.
This script measures it DIRECTLY, converting "candidate mechanism" into evidence.

WHAT IT MEASURES
----------------
YOLOMG (YOLOv5) emits predictions at three detection heads:
    P3 (stride  8)  -> small  objects
    P4 (stride 16)  -> medium objects
    P5 (stride 32)  -> large  objects
A ground-truth box is assigned (by size/anchor matching) to the head whose
scale it matches, so the box/positive-objectness gradient for large targets
flows into the P5 head. If a training distribution has no large targets, the
P5 head receives no positive gradient -> starvation.

We load the Stage 2 best checkpoint (the exact model entering Stage 3), run
forward+backward (NO optimizer step, NO weight change) on:
    (1) CST train batches          -- the Stage 3 distribution (0% large)
    (2) Anti-UAV-RGBT val batches  -- the T1 distribution (has large targets)
    (3) Anti-UAV-RGBT, large-only  -- control: only large GT kept (optional)
and record the L2 norm of the gradient reaching each detection head, averaged
over many batches (mean +/- std).

EXPECTED RESULT (the thing to put in the thesis)
------------------------------------------------
P5 (large) head gradient under CST  ~=  0
P5 (large) head gradient under RGBT  >>  0
i.e. the large-target gradient signal exists when large targets are present and
vanishes under the CST distribution -- direct evidence of gradient starvation.

COST
----
No training. ~num_batches forward+backward passes on a single GPU. Minutes.
One GPU is plenty (DDP not used).

USAGE
-----
  python grad_starvation_diagnostic.py \
      --weights   /projects/prjs2041/runs/stage2/seed42/weights/best.pt \
      --num-batches 60 \
      --control-only-large \
      --out       /projects/prjs2041/runs/diagnostics/grad_starvation

Run it once per Stage 2 seed checkpoint (42/123/999) if you want a spread.
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Reuse the EXACT model construction, loss, datasets and helpers from Stage 3.
# Importing train_stage3 also runs its sys.path inserts so YOLOMG is importable.
import train_stage3 as s3


# ──────────────────────────────────────────────────────────────────────────────
def build_model(weights: Path, device):
    """Load the Stage 2 checkpoint into YOLOMG exactly as train_stage3 does."""
    ckpt  = torch.load(str(weights), map_location=device, weights_only=False)
    model = s3.Model(s3.MODEL_CFG, ch=3, ch2=3, nc=s3.NC).to(device)
    model.nc = s3.NC
    model.hyp = s3.HYP
    model.names = s3.NAMES

    state_src = ckpt.get('ema') or ckpt.get('model')      # prefer EMA weights
    sd = state_src.float().state_dict() if state_src else {}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  WARNING: {len(missing)} missing keys when loading weights')

    # Train mode so Detect returns the raw training output ComputeLoss expects,
    # but freeze BatchNorm so running statistics are NOT mutated by the diagnostic
    # (we measure gradients on the incoming model; we never want to change it).
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return model


def filter_targets_by_stratum(targets: torch.Tensor, keep: set, imgsz: int):
    """
    targets: [N, 6] = [batch_idx, cls, x, y, w, h] (normalised).
    Keep only rows whose size stratum is in `keep`. Mirrors _size_cat in
    train_stage3 (w_px = w*imgsz, h_px = h*imgsz).
    """
    if targets.numel() == 0:
        return targets
    keep_rows = []
    for row in targets:
        w_px = float(row[4]) * imgsz
        h_px = float(row[5]) * imgsz
        if s3._size_cat(w_px, h_px) in keep:
            keep_rows.append(row)
    if not keep_rows:
        return targets.new_zeros((0, 6))
    return torch.stack(keep_rows, 0)


def head_grad_norms(model):
    """Per-detection-head gradient L2 norm. Returns dict head_idx -> float."""
    detect = s3.de_parallel(model).model[-1]      # YOLOv5 Detect module
    out = {}
    for j in range(detect.nl):                    # nl = 3 (P3, P4, P5)
        sq = 0.0
        for p in detect.m[j].parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        out[j] = sq ** 0.5
    return out


def total_trainable_grad_norm(model):
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return sq ** 0.5


# ──────────────────────────────────────────────────────────────────────────────
@torch.enable_grad()
def measure(model, loader, device, compute_loss, n_batches,
            keep_stratum=None, imgsz=640, tag=''):
    """
    Run n_batches forward+backward passes, recording per-head gradient norms.
    keep_stratum: optional set of stratum names to keep in the GT (e.g. {'large'}).
    Returns: dict with arrays of per-head norms, total norm, and loss components.
    """
    detect = s3.de_parallel(model).model[-1]
    nl = detect.nl
    per_head = {j: [] for j in range(nl)}
    totals, lboxs, lobjs, lclss, ntargets = [], [], [], [], []

    it = iter(loader)
    done = 0
    while done < n_batches:
        try:
            imgs, imgs2, targets, _ = next(it)
        except StopIteration:
            break

        if keep_stratum is not None:
            targets = filter_targets_by_stratum(targets, keep_stratum, imgsz)
            if len(targets) == 0:
                continue                          # skip batches with no kept GT

        imgs    = imgs.to(device, non_blocking=True).float()
        imgs2   = imgs2.to(device, non_blocking=True).float()
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)
        pred = model(imgs, imgs2)                 # train mode -> raw head outputs
        loss, items = compute_loss(pred, targets) # items = [lbox, lobj, lcls]
        loss.backward()

        hn = head_grad_norms(model)
        for j in range(nl):
            per_head[j].append(hn[j])
        totals.append(total_trainable_grad_norm(model))
        items = items.detach().cpu().numpy()
        lboxs.append(float(items[0])); lobjs.append(float(items[1]))
        lclss.append(float(items[2])); ntargets.append(int(len(targets)))
        done += 1

    model.zero_grad(set_to_none=True)
    print(f'  [{tag}] measured {done} batches '
          f'(mean GT/batch = {np.mean(ntargets):.1f})')
    return {
        'per_head': {j: np.array(v) for j, v in per_head.items()},
        'total':    np.array(totals),
        'lbox':     np.array(lboxs),
        'lobj':     np.array(lobjs),
        'lcls':     np.array(lclss),
        'nbatches': done,
    }


def msd(a):
    return (float(np.mean(a)), float(np.std(a))) if len(a) else (float('nan'), float('nan'))


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Gradient starvation diagnostic (Stage 2 -> Stage 3).')
    ap.add_argument('--weights', type=str, default=str(s3.DEFAULT_STAGE2_WEIGHTS),
                    help='Stage 2 checkpoint (best.pt) to probe.')
    ap.add_argument('--num-batches', type=int, default=60)
    ap.add_argument('--batch-size',  type=int, default=16)
    ap.add_argument('--workers',     type=int, default=4)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--device',      type=str, default='')
    ap.add_argument('--cst-root',    type=str, default=str(s3.CST_ROOT))
    ap.add_argument('--rgbt-root',   type=str, default=str(s3.RGBT_ROOT))
    ap.add_argument('--control-only-large', action='store_true',
                    help='Add a 3rd condition: RGBT batches with only large GT kept.')
    ap.add_argument('--out', type=str, default='/projects/prjs2041/runs/diagnostics/grad_starvation')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = s3.select_device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nGradient starvation diagnostic')
    print(f'  weights : {args.weights}')
    print(f'  device  : {device}')
    print(f'  batches : {args.num_batches} x bs {args.batch_size}\n')

    model = build_model(Path(args.weights), device)
    compute_loss = s3.ComputeLoss(s3.de_parallel(model))

    detect = s3.de_parallel(model).model[-1]
    strides = [int(x) for x in detect.stride.tolist()]
    head_label = {j: f'P{j+3} (stride {strides[j]})' for j in range(detect.nl)}
    scale_name = {0: 'small', 1: 'medium', 2: 'large'}

    # ── Datasets (reuse Stage 3 wrappers; img2 = zeros, same letterboxing) ──
    cst   = s3.Stage3Dataset(Path(args.cst_root),  split='train', imgsz=s3.IMG_SIZE)
    rgbt  = s3.RGBTValDataset(Path(args.rgbt_root),                imgsz=s3.IMG_SIZE)

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.workers, pin_memory=True,
            collate_fn=s3._collate, drop_last=True)

    print('Measuring CST (Stage 3 distribution, 0% large) ...')
    res_cst = measure(model, make_loader(cst, True), device, compute_loss,
                      args.num_batches, tag='CST')

    print('Measuring Anti-UAV-RGBT (T1 distribution, has large) ...')
    res_rgbt = measure(model, make_loader(rgbt, True), device, compute_loss,
                       args.num_batches, tag='RGBT')

    res_large = None
    if args.control_only_large:
        print('Measuring Anti-UAV-RGBT, LARGE-ONLY control ...')
        res_large = measure(model, make_loader(rgbt, True), device, compute_loss,
                            args.num_batches, keep_stratum={'large'},
                            imgsz=s3.IMG_SIZE, tag='RGBT-large-only')

    # ── Report ──────────────────────────────────────────────────────────────
    conditions = [('CST (Stage 3, 0% large)', res_cst),
                  ('RGBT (T1, all strata)',   res_rgbt)]
    if res_large is not None:
        conditions.append(('RGBT (large GT only)', res_large))

    print('\n' + '=' * 78)
    print('PER-HEAD GRADIENT NORM  (mean +/- std over batches)')
    print('=' * 78)
    header = f'{"head":<18}' + ''.join(f'{c[0]:>26}' for c in conditions)
    print(header)
    for j in range(detect.nl):
        line = f'{head_label[j]+" ["+scale_name[j]+"]":<18}'
        for _, res in conditions:
            m, s = msd(res['per_head'][j])
            line += f'{m:>14.4f} +/-{s:>7.4f}'
        print(line)
    # total norm row
    line = f'{"TOTAL (all params)":<18}'
    for _, res in conditions:
        m, s = msd(res['total'])
        line += f'{m:>14.4f} +/-{s:>7.4f}'
    print(line)
    print('=' * 78)

    # Headline ratio
    p5_cst,  _ = msd(res_cst['per_head'][detect.nl - 1])
    p5_rgbt, _ = msd(res_rgbt['per_head'][detect.nl - 1])
    ratio = (p5_rgbt / p5_cst) if p5_cst > 0 else float('inf')
    print(f'\nLarge-head (P5) gradient:  RGBT / CST = {ratio:.1f}x'
          f'   (CST={p5_cst:.4f}, RGBT={p5_rgbt:.4f})')
    print('Near-zero P5 gradient under CST = direct evidence of starvation of '
          'the large-target pathway.\n')

    # ── CSV (per-condition, per-head means + std) ────────────────────────────
    csv_path = out_dir / 'grad_starvation_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'head', 'scale', 'stride',
                    'grad_norm_mean', 'grad_norm_std', 'n_batches',
                    'lbox_mean', 'lobj_mean', 'lcls_mean'])
        for cname, res in conditions:
            for j in range(detect.nl):
                m, s = msd(res['per_head'][j])
                w.writerow([cname, f'P{j+3}', scale_name[j], strides[j],
                            f'{m:.6f}', f'{s:.6f}', res['nbatches'],
                            f'{np.mean(res["lbox"]):.6f}',
                            f'{np.mean(res["lobj"]):.6f}',
                            f'{np.mean(res["lcls"]):.6f}'])
    print(f'Saved summary: {csv_path}')


if __name__ == '__main__':
    main()
