#!/usr/bin/env python3
"""
eval_stage1.py
--------------
Standalone evaluation of a Stage 1 checkpoint (best.pt) on the
Anti-UAV-RGBT validation set.

Computes REAL mAP@0.5 and mAP@0.5:0.95 using 10 IoU thresholds
(0.50, 0.55, 0.60, ..., 0.95), unlike the training-time metric which
used a single threshold due to the ap_per_class unsqueeze(1) fix.

Usage (single GPU, no torchrun needed):
  python eval_stage1.py \
      --weights /projects/prjs2041/runs/stage1/antiuav_rgbt14/weights/best.pt \
      --dataset-root $TMPDIR/Anti-UAV-RGBT

Outputs:
  Prints mAP@0.5 and mAP@0.5:0.95 to stdout.
  Writes results to <weights_dir>/eval_results.txt
"""

import sys
import os
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT  = Path('/projects/prjs2041/YOLOMG')
UAV_CODE     = Path('/projects/prjs2041/uav_code')

sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

from models.yolo import Model
from utils.general import non_max_suppression, xywh2xyxy
from utils.metrics import ap_per_class, box_iou
from utils.torch_utils import select_device

from datasets import AntiUAVRGBTDataset

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE   = 640
BATCH_SIZE = 32      # single GPU — can go higher than training BS
WORKERS    = 4
NC         = 1
NAMES      = {0: 'UAV'}
CONF_THRES = 0.001
IOU_THRES  = 0.6     # NMS IoU threshold
IOV_V      = torch.linspace(0.5, 0.95, 10)  # 10 eval IoU thresholds


# ══════════════════════════════════════════════════════════════════════════════
# Dataset  (same as Stage 1 training)
# ══════════════════════════════════════════════════════════════════════════════

def _letterbox_tensor(img: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = img.shape
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = F.interpolate(img.unsqueeze(0), size=(nh, nw),
                        mode='bilinear', align_corners=False).squeeze(0)
    pad_h, pad_w = size - nh, size - nw
    img = F.pad(img, (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2), value=0.5)
    return img


class ValDataset(Dataset):
    def __init__(self, root: Path, imgsz: int = 640):
        self.ds    = AntiUAVRGBTDataset(root=root, split='val')
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = _letterbox_tensor(sample['image'], self.imgsz)
        img2   = torch.zeros_like(img)   # no motion masks at Stage 1
        labels = sample['labels']
        meta   = sample['meta']
        path   = f"{meta.get('seq','seq')}_{meta.get('frame',idx)}"
        return img, img2, labels, path

    @staticmethod
    def collate_fn(batch):
        imgs, imgs2, labels_list, paths = zip(*batch)
        imgs  = torch.stack(imgs)
        imgs2 = torch.stack(imgs2)
        targets = []
        for i, lbl in enumerate(labels_list):
            if len(lbl) > 0:
                lbl_t = torch.as_tensor(lbl, dtype=torch.float32)
                bi    = torch.full((len(lbl_t), 1), float(i))
                targets.append(torch.cat([bi, lbl_t], dim=1))
        targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
        return imgs, imgs2, targets, list(paths)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation with 10 IoU thresholds
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model, loader, device, imgsz=640):
    """
    Evaluate model over the full val set using 10 IoU thresholds.

    correct has shape (N, 10): correct[i, j] = True if detection i
    is a true positive at IoU threshold iou_v[j].

    Returns list of (correct, conf, pred_cls, target_cls) per image.
    """
    iou_v = IOV_V.to(device)
    model.eval()
    stats = []

    n_batches = len(loader)
    for bi, (imgs, imgs2, targets, _) in enumerate(loader):
        imgs    = imgs.to(device)
        imgs2   = imgs2.to(device)
        targets = targets.to(device)

        preds = model(imgs, imgs2)
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = non_max_suppression(preds, CONF_THRES, IOU_THRES)

        for si, det in enumerate(preds):
            gt   = targets[targets[:, 0] == si, 1:]   # (nl, 5) cls xywh
            nl   = len(gt)
            tcls = gt[:, 0].long().tolist() if nl else []

            if len(det) == 0:
                if nl:
                    stats.append((
                        torch.zeros(0, 10, dtype=torch.bool),
                        torch.zeros(0),
                        torch.zeros(0),
                        tcls,
                    ))
                continue

            # correct[i, j] = True if detection i is TP at threshold j
            correct = torch.zeros(len(det), 10, dtype=torch.bool, device=device)

            if nl:
                tbox     = xywh2xyxy(gt[:, 1:5]) * imgsz   # de-normalise to pixels
                iou_mat  = box_iou(tbox, det[:, :4])        # (nl, nd)

                for j, iou_th in enumerate(iou_v):
                    x = torch.where(iou_mat >= iou_th)
                    if x[0].shape[0] == 0:
                        # Thresholds are ascending — no point checking higher ones
                        break
                    # Build (gt_idx, det_idx, iou) and do greedy matching
                    matches = torch.cat((
                        torch.stack(x, 1).float(),
                        iou_mat[x[0], x[1]].unsqueeze(1)
                    ), 1).cpu().numpy()   # (M, 3)
                    if matches.shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1],
                                                    return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0],
                                                    return_index=True)[1]]
                    correct[matches[:, 1].astype(int), j] = True

            stats.append((
                correct.cpu(),       # (nd, 10)
                det[:, 4].cpu(),     # conf
                det[:, 5].cpu(),     # pred class
                tcls,                # target classes
            ))

        if (bi + 1) % 50 == 0:
            print(f'  [{bi+1:4d}/{n_batches}]', flush=True)

    model.train()
    return stats


def compute_map(stats):
    if not stats:
        return 0.0, 0.0
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        _, _, _, _, _, ap, _ = ap_per_class(
            *stats, plot=False, names=NAMES
        )
        # ap shape: (num_classes, num_iou_thresholds)
        map50 = float(ap[:, 0].mean()) if ap.ndim == 2 else float(ap[0])
        mapxx = float(ap.mean())       if ap.ndim == 2 else float(ap.mean())
    else:
        map50, mapxx = 0.0, 0.0
    return map50, mapxx


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',      type=str,
                   default='/projects/prjs2041/runs/stage1/antiuav_rgbt14/weights/best.pt')
    p.add_argument('--dataset-root', type=str,
                   default='/gpfs/scratch1/shared/knguyen1/Anti-UAV-RGBT',
                   help='Path to Anti-UAV-RGBT root (copy to scratch first)')
    p.add_argument('--batch-size',   type=int, default=BATCH_SIZE)
    p.add_argument('--workers',      type=int, default=WORKERS)
    p.add_argument('--device',       type=str, default='0')
    p.add_argument('--imgsz',        type=int, default=IMG_SIZE)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    weights = Path(args.weights)
    print(f'Loading checkpoint: {weights}')
    ckpt  = torch.load(weights, map_location=device, weights_only=False)

    # Prefer EMA weights (smoother, better for eval); fall back to model
    if 'ema' in ckpt and ckpt['ema'] is not None:
        model = ckpt['ema'].float().to(device)
        print('  Using EMA weights')
    else:
        model = ckpt['model'].float().to(device)
        print('  Using model weights (no EMA found)')

    model.nc    = NC
    model.names = list(NAMES.values())
    model.eval()

    epoch = ckpt.get('epoch', '?')
    print(f'  Checkpoint epoch: {epoch}')
    print(f'  Reported best_fitness (mAP@0.5 during training): '
          f'{ckpt.get("best_fitness", "?"):.4f}')

    # ── Val dataset ───────────────────────────────────────────────────────────
    dataset_root = Path(args.dataset_root)
    print(f'\nLoading val set from {dataset_root} …')
    val_ds     = ValDataset(dataset_root, imgsz=args.imgsz)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=ValDataset.collate_fn,
    )
    print(f'  {len(val_ds):,} frames  |  {len(val_loader)} batches')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f'\nRunning evaluation (10 IoU thresholds: 0.50 → 0.95) …')
    stats       = evaluate_full(model, val_loader, device, args.imgsz)
    map50, mapxx = compute_map(stats)

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 50)
    print(f'  mAP@0.5          : {map50:.4f}')
    print(f'  mAP@0.5:0.95     : {mapxx:.4f}')
    print('=' * 50)

    # Save alongside the weights
    out_txt = weights.parent / 'eval_results.txt'
    out_txt.write_text(
        f'checkpoint : {weights}\n'
        f'epoch      : {epoch}\n'
        f'val_frames : {len(val_ds):,}\n'
        f'mAP@0.5    : {map50:.6f}\n'
        f'mAP@0.5:0.95: {mapxx:.6f}\n'
    )
    print(f'\nResults written to {out_txt}')

    # Also update stage1_mAP.txt with the authoritative value
    maptxt = weights.parent.parent / 'stage1_mAP.txt'
    maptxt.write_text(f'{map50:.6f}\n')
    print(f'T1 mAP written to  {maptxt}  ← used for Forgetting Measure')


if __name__ == '__main__':
    main()
