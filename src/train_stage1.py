#!/usr/bin/env python3
"""
train_stage1.py
---------------
Stage 1: Train YOLOMG on Anti-UAV-RGBT IR stream (full supervision).

This is the first step of the three-stage continual learning curriculum:
  Stage 1  →  Establishes T1 mAP ceiling on AntiUAVRGBT IR
  Stage 2  →  Teacher-Student UDA to AntiUAV410
  Stage 3  →  Continual fine-tuning on CST (Scale-Stratified Herding)

The Forgetting Measure is computed relative to the mAP produced here:
  FM = mAP_T1_after_T2 − mAP_T1_after_T1   (negative = forgetting)

Model: YOLOMG (dual-input YOLOv5 variant)
  img1 : IR appearance frame  — (B, 3, H, W) float32 [0, 1]
  img2 : Motion mask          — zeros for Stage 1 (no precomputed masks
                                for AntiUAVRGBT; motion branch trains on
                                zero signal and contributes nothing, which
                                is correct since AntiUAVRGBT has no ego-
                                motion unlike ARD100).

Usage (direct, testing only — use sbatch for real runs):
  python3 train_stage1.py

SLURM:
  sbatch run_stage1.sh

Outputs:
  /projects/prjs2041/runs/stage1/<timestamp>/
      weights/last.pt
      weights/best.pt
      results.csv
      stage1_mAP.txt   ← final mAP@0.5 written here for FM computation
"""

import sys
import os
import time
import csv
import math
import yaml
import random
import logging
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT  = Path('/projects/prjs2041/YOLOMG')
UAV_CODE     = Path('/projects/prjs2041/uav_code')
DATASET_ROOT = Path('/projects/prjs2041/datasets/Anti-UAV-RGBT')
FRAMES_ROOT  = None   # set via --frames-root to use pre-extracted JPEGs
SAVE_ROOT    = Path('/projects/prjs2041/runs/stage1')
LOGS_DIR     = Path('/projects/prjs2041/logs')

sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

# ── YOLOMG imports ─────────────────────────────────────────────────────────────
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import (
    colorstr, increment_path, init_seeds, check_img_size,
    non_max_suppression, xywh2xyxy
)
from utils.torch_utils import select_device, ModelEMA, de_parallel, EarlyStopping
from utils.metrics import ap_per_class, box_iou

# ── Our dataset ────────────────────────────────────────────────────────────────
from datasets import AntiUAVRGBTDataset, AntiUAVRGBTFramesDataset

# ── Hyperparameters (match thesis methodology) ────────────────────────────────
HYP = {
    'lr0':             1e-2,    # initial LR
    'lrf':             1e-2,    # final LR ratio (cosine target = lr0 * lrf)
    'momentum':        0.937,
    'weight_decay':    5e-4,
    'warmup_epochs':   3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr':  0.1,
    'box':             0.05,
    'cls':             0.5,
    'cls_pw':          1.0,
    'obj':             1.0,
    'obj_pw':          1.0,
    'iou_t':           0.20,
    'anchor_t':        4.0,
    'fl_gamma':        0.0,
    'label_smoothing': 0.0,
    # Augmentation — keep minimal for IR (no colour jitter)
    'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.2,
    'degrees': 0.0, 'translate': 0.1, 'scale': 0.3,
    'shear': 0.0, 'perspective': 0.0,
    'flipud': 0.0, 'fliplr': 0.5,
    'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
}

# ── Fixed config ───────────────────────────────────────────────────────────────
IMG_SIZE   = 640          # must be multiple of 32
EPOCHS     = 100
BATCH_SIZE = 64           # 16 per GPU × 4 GPUs; override via --batch-size
WORKERS    = 16           # 4 per GPU × 4 GPUs
SEED       = 42
NC         = 1            # one class: UAV
NAMES      = ['UAV']
MODEL_CFG  = str(YOLOMG_ROOT / 'models' / 'dual_uav2.yaml')

# NMS thresholds for evaluation
CONF_THRES = 0.001
IOU_THRES  = 0.6
MAP_IOU    = 0.5          # IoU threshold for mAP@0.5


# ══════════════════════════════════════════════════════════════════════════════
# Dataset wrapper
# ══════════════════════════════════════════════════════════════════════════════

class Stage1Dataset(Dataset):
    """
    Wraps AntiUAVRGBTDataset for YOLOMG.

    Returns:
        img1   (3, H, W) float [0,1]  — IR frame
        img2   (3, H, W) float [0,1]  — zeros (no motion masks at Stage 1)
        labels (N, 5)    float        — [cls, xc, yc, w, h] normalised
        path   str                    — identifier for logging
    """

    def __init__(self, root: Path, split: str, imgsz: int = 640,
                 frames_root: Path = None):
        if frames_root is not None:
            # Fast path: pre-extracted JPEGs (eliminates VideoCapture random seeks)
            self.ds = AntiUAVRGBTFramesDataset(
                frames_root=frames_root, ann_root=root, split=split)
        else:
            self.ds = AntiUAVRGBTDataset(root=root, split=split)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = sample['image']    # (3, H, W) float [0, 1]
        labels = sample['labels']   # (N, 5) [cls, xc, yc, w, h]
        meta   = sample['meta']

        # Letterbox resize to fixed square (preserve aspect ratio with padding)
        img = _letterbox_tensor(img, self.imgsz)

        # Motion mask: zeros for Stage 1
        img2 = torch.zeros_like(img)

        path = f"{meta.get('seq', 'seq')}_{meta.get('frame', idx)}"
        return img, img2, labels, path

    @staticmethod
    def collate_fn(batch):
        imgs, imgs2, labels_list, paths = zip(*batch)
        imgs  = torch.stack(imgs)   # (B, 3, H, W)
        imgs2 = torch.stack(imgs2)  # (B, 3, H, W)

        # Build YOLO-format targets: (N, 6) [batch_idx, cls, xc, yc, w, h]
        targets = []
        for i, lbl in enumerate(labels_list):
            if len(lbl) > 0:
                lbl_t    = torch.as_tensor(lbl, dtype=torch.float32)
                bi       = torch.full((len(lbl_t), 1), float(i))
                targets.append(torch.cat([bi, lbl_t], dim=1))
        targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))

        # Pad to 7-tuple so YOLOMG val.run is compatible if needed later
        return imgs, imgs2, targets, list(paths), list(paths), None, None


def _letterbox_tensor(img: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (3, H, W) float tensor to (3, size, size) with letterboxing."""
    _, h, w = img.shape
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = F.interpolate(img.unsqueeze(0), size=(nh, nw),
                        mode='bilinear', align_corners=False).squeeze(0)
    # Pad to square with 0.5 (mid-grey, common YOLO convention)
    pad_h = size - nh
    pad_w = size - nw
    top, left = pad_h // 2, pad_w // 2
    img = F.pad(img, (left, pad_w - left, top, pad_h - top), value=0.5)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation  (mAP@0.5)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, imgsz=640):
    """
    Collect per-image detection stats on the given loader (one rank's shard).

    Returns:
        stats : list of (correct, conf, pred_cls, target_cls) tuples — one per image.
                Pass to compute_map() to get mAP values.  Collecting raw stats
                instead of aggregated mAP lets us gather across DDP ranks before
                computing the final metric (see training loop).
    """
    model.eval()
    stats = []   # list of (correct, conf, pred_cls, target_cls) per image

    for imgs, imgs2, targets, *_ in loader:
        imgs   = imgs.to(device)
        imgs2  = imgs2.to(device)
        targets = targets.to(device)

        preds = model(imgs, imgs2)
        # YOLOMG eval mode returns (inference_out, train_out) tuple
        if isinstance(preds, tuple):
            preds = preds[0]             # take decoded inference output
        preds = non_max_suppression(preds, CONF_THRES, IOU_THRES)  # list[B] of (n,6)

        for si, det in enumerate(preds):
            gt   = targets[targets[:, 0] == si, 1:]   # (nl, 5) cls xywh
            nl   = len(gt)
            tcls = gt[:, 0].long().tolist() if nl else []

            if len(det) == 0:
                if nl:
                    stats.append((
                        torch.zeros(0, 1, dtype=torch.bool),  # (0,1): 2-D for ap_per_class
                        torch.zeros(0), torch.zeros(0), tcls
                    ))
                continue

            # Build predicted boxes in pixel coords (xyxy)
            predn = det.clone()   # (n, 6) xyxy conf cls

            # Build GT boxes in pixel coords (xyxy)
            if nl:
                tbox = xywh2xyxy(gt[:, 1:5])
                tbox = tbox * imgsz          # de-normalise
                tbox = tbox.to(device)

                iou = box_iou(tbox, predn[:, :4])   # (nl, n)
                # For each GT, find the best-matching prediction
                correct = torch.zeros(len(det), dtype=torch.bool, device=device)
                if iou.numel():
                    x = torch.where((iou >= MAP_IOU))
                    if x[0].shape[0]:
                        matches = torch.cat((
                            torch.stack(x, 1),
                            iou[x[0], x[1]][:, None]
                        ), 1).cpu().numpy()  # [gt_idx, pred_idx, iou]
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int)] = True
            else:
                correct = torch.zeros(len(det), dtype=torch.bool)

            stats.append((
                correct.cpu().unsqueeze(1),  # (N,) → (N,1): 2-D for ap_per_class
                det[:, 4].cpu(),
                det[:, 5].cpu(),
                tcls
            ))

    model.train()
    return stats


def compute_map(stats):
    """
    Compute mAP@0.5 and mAP@0.5:0.95 from aggregated stats.

    Args:
        stats : list of (correct, conf, pred_cls, target_cls) — can be the merged
                list from multiple ranks.

    Returns:
        map50 : float
        mapxx : float
    """
    if not stats:
        return 0.0, 0.0

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        _, _, _, _, _, ap, _ = ap_per_class(*stats, plot=False, names={0: 'UAV'})  # returns tp,fp,p,r,f1,ap,cls
        map50 = float(ap[:, 0].mean()) if ap.ndim == 2 else float(ap[0])
        mapxx = float(ap.mean())       if ap.ndim == 2 else float(ap.mean())
    else:
        map50, mapxx = 0.0, 0.0

    return map50, mapxx


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(save_dir: Path, device: torch.device, rank: int = -1, world_size: int = 1):
    """
    rank       : -1 = single-GPU (no DDP), 0..N-1 = DDP rank
    world_size : total number of DDP processes (1 = single-GPU)
    """
    is_main = rank in (-1, 0)   # only rank-0 logs / saves / evaluates

    init_seeds(SEED + rank if rank != -1 else SEED)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(exist_ok=True)

    # ── Logging (rank 0 only) ─────────────────────────────────────────────────
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.handlers = []   # clear any handlers set by YOLOMG imports
    if is_main:
        log.addHandler(logging.FileHandler(save_dir / 'train.log'))
    log.addHandler(logging.StreamHandler(sys.stdout))

    if is_main:
        log.info(f'Stage 1 training  →  {save_dir}')
        log.info(f'Device: {device}  |  IMG_SIZE={IMG_SIZE}  EPOCHS={EPOCHS}  '
                 f'BS={BATCH_SIZE}  world_size={world_size}')

    # ── Save config (rank 0 only) ─────────────────────────────────────────────
    if is_main:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(HYP, f)

    # ── Datasets ──────────────────────────────────────────────────────────────
    if is_main:
        log.info('Loading datasets …')
    fr = Path(FRAMES_ROOT) if FRAMES_ROOT else None
    if fr and is_main:
        log.info(f'  Using pre-extracted JPEG frames from {fr}')
    train_ds = Stage1Dataset(DATASET_ROOT, split='train', imgsz=IMG_SIZE, frames_root=fr)
    val_ds   = Stage1Dataset(DATASET_ROOT, split='val',   imgsz=IMG_SIZE, frames_root=fr)
    if is_main:
        log.info(f'  train: {len(train_ds):,} frames  |  val: {len(val_ds):,} frames')

    # DDP: each rank sees a non-overlapping shard of both training and val data.
    # Training: DistributedSampler shuffles per epoch (set_epoch called in loop).
    # Validation: DistributedSampler splits val set across ranks (no shuffle).
    #   Each rank collects its own stats; rank 0 gathers all and computes mAP.
    #   This cuts validation time from ~40 min (rank 0 alone) to ~10 min (4× parallel).
    train_sampler = (DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True)
                     if rank != -1 else None)
    val_sampler = (DistributedSampler(val_ds, num_replicas=world_size,
                                      rank=rank, shuffle=False)
                   if rank != -1 else None)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),   # DistributedSampler handles shuffle
        sampler=train_sampler,
        num_workers=WORKERS,
        pin_memory=True,
        collate_fn=Stage1Dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        sampler=val_sampler,
        num_workers=WORKERS, pin_memory=True,
        collate_fn=Stage1Dataset.collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if is_main:
        log.info(f'Building YOLOMG from {MODEL_CFG} …')
    model = Model(MODEL_CFG, ch=3, ch2=3, nc=NC).to(device)
    model.nc    = NC
    model.hyp   = HYP
    model.names = NAMES
    gs = max(int(model.stride.max()), 32)

    if is_main:
        log.info(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Wrap with DDP — must come AFTER setting .nc / .hyp / .names so those
    # attrs survive on model.module (de_parallel will unwrap as needed).
    if rank != -1:
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Split params: BN weights (no decay) | weights (decay) | biases
    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    optimizer = SGD(g0, lr=HYP['lr0'], momentum=HYP['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': HYP['weight_decay']})
    optimizer.add_param_group({'params': g2})
    del g0, g1, g2

    # Cosine LR schedule
    lf = lambda x: ((1 - math.cos(x * math.pi / EPOCHS)) / 2) * (HYP['lrf'] - 1) + 1
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    # ── AMP + EMA ─────────────────────────────────────────────────────────────
    scaler  = amp.GradScaler(enabled=(device.type != 'cpu'))
    ema     = ModelEMA(model)
    stopper = EarlyStopping(patience=30)

    # ── Loss ──────────────────────────────────────────────────────────────────
    compute_loss = ComputeLoss(de_parallel(model))   # always unwrap for ComputeLoss

    # ── Results CSV (rank 0 only) ─────────────────────────────────────────────
    csv_path = save_dir / 'results.csv'
    if is_main:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss_box', 'loss_obj', 'loss_cls',
                             'loss_total', 'mAP50', 'mAP50-95', 'lr'])

    # ── Training loop ─────────────────────────────────────────────────────────
    nb           = len(train_loader)
    nw           = max(round(HYP['warmup_epochs'] * nb), 100)
    best_fitness = 0.0
    last_path    = save_dir / 'weights' / 'last.pt'
    best_path    = save_dir / 'weights' / 'best.pt'

    if is_main:
        log.info(f'\nStarting training for {EPOCHS} epochs …\n')

    for epoch in range(EPOCHS):
        model.train()
        # DDP: tell sampler which epoch we're on (controls per-epoch shuffle seed)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        mloss = torch.zeros(3, device=device)   # box, obj, cls

        for i, (imgs, imgs2, targets, *_) in enumerate(train_loader):
            ni   = i + nb * epoch
            imgs  = imgs.to(device,  non_blocking=True)
            imgs2 = imgs2.to(device, non_blocking=True)
            targets = targets.to(device)

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(
                        ni, xi,
                        [HYP['warmup_bias_lr'] if j == 2 else 0.0,
                         x['initial_lr'] * lf(epoch)]
                    )
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [HYP['warmup_momentum'], HYP['momentum']]
                        )

            # Forward + loss
            with amp.autocast(enabled=(device.type != 'cpu')):
                pred = model(imgs, imgs2)
                loss, loss_items = compute_loss(pred, targets)

            # Backward + optimise
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

            mloss = (mloss * i + loss_items) / (i + 1)

            if i % 50 == 0 and is_main:
                mem = f'{torch.cuda.memory_reserved() / 1e9:.2f}G' \
                      if torch.cuda.is_available() else 'cpu'
                log.info(
                    f'  [{epoch:3d}/{EPOCHS-1}][{i:4d}/{nb}]  '
                    f'mem={mem}  '
                    f'box={mloss[0]:.4f}  obj={mloss[1]:.4f}  cls={mloss[2]:.4f}'
                )

        scheduler.step()

        # ── Aggregate losses across DDP ranks ────────────────────────────────
        if rank != -1:
            dist.all_reduce(mloss, op=dist.ReduceOp.SUM)
            mloss /= world_size

        # ── End-of-epoch validation (all ranks participate) ──────────────────
        # Each rank evaluates its shard of the val set (via val_sampler).
        # Raw stats are gathered to rank 0, which computes the final mAP.
        # This is ~4× faster than letting only rank 0 process the full val set,
        # and avoids NCCL timeout while ranks 1-3 sit idle for ~40 min.
        local_stats = evaluate(ema.ema, val_loader, device, IMG_SIZE)

        map50, mapxx = 0.0, 0.0
        if rank != -1:
            # all_gather_object: each rank sends its stats list to every rank,
            # but only rank 0 will use the merged result.
            all_stats_list = [None] * world_size
            dist.all_gather_object(all_stats_list, local_stats)
            if is_main:
                merged_stats = []
                for s in all_stats_list:
                    merged_stats.extend(s)
                map50, mapxx = compute_map(merged_stats)
        else:
            # Single-GPU path
            map50, mapxx = compute_map(local_stats)

        if is_main:
            lr_now = optimizer.param_groups[0]['lr']
            total_loss = mloss.sum().item()

            log.info(
                f'\nEpoch {epoch:3d}/{EPOCHS-1}  '
                f'loss={total_loss:.4f}  mAP@0.5={map50:.4f}  mAP@0.5:0.95={mapxx:.4f}\n'
            )

            # CSV
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch, mloss[0].item(), mloss[1].item(), mloss[2].item(),
                    total_loss, map50, mapxx, lr_now
                ])

            # Checkpoint
            ckpt = {
                'epoch':        epoch,
                'best_fitness': best_fitness,
                'model':        deepcopy(de_parallel(model)).half(),
                'ema':          deepcopy(ema.ema).half(),
                'updates':      ema.updates,
                'optimizer':    optimizer.state_dict(),
                'hyp':          HYP,
                'date':         datetime.now().isoformat(),
            }
            torch.save(ckpt, last_path)
            if map50 > best_fitness:
                best_fitness = map50
                torch.save(ckpt, best_path)
                log.info(f'  ✓ New best mAP@0.5 = {best_fitness:.4f}  (saved {best_path})')

        # ── Early stopping — rank 0 decides, all ranks obey ──────────────────
        stop = False
        if is_main:
            stop = stopper(epoch=epoch, fitness=map50)
            if stop:
                log.info('Early stopping triggered.')
        if rank != -1:
            # Broadcast the stop flag from rank 0 to all other ranks
            stop_t = torch.tensor(int(stop), device=device)
            dist.broadcast(stop_t, src=0)
            stop = bool(stop_t.item())
        if stop:
            break

    # ── Final report (rank 0 only) ────────────────────────────────────────────
    if is_main:
        log.info(f'\n{"=" * 60}')
        log.info(f'Stage 1 complete.')
        log.info(f'Best mAP@0.5 = {best_fitness:.4f}')
        log.info(f'Weights: {best_path}')
        log.info(f'Results: {csv_path}')

        # Write T1 mAP for FM computation in later stages
        maptxt = save_dir / 'stage1_mAP.txt'
        maptxt.write_text(f'{best_fitness:.6f}\n')
        log.info(f'T1 mAP written to {maptxt}  ← used for Forgetting Measure')

    # ── Cleanup DDP ──────────────────────────────────────────────────────────
    if rank != -1:
        dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Stage 1: YOLOMG on AntiUAVRGBT')
    p.add_argument('--epochs',       type=int, default=EPOCHS)
    p.add_argument('--batch-size',   type=int, default=BATCH_SIZE)
    p.add_argument('--imgsz',        type=int, default=IMG_SIZE)
    p.add_argument('--workers',      type=int, default=WORKERS)
    p.add_argument('--device',       type=str, default='')
    p.add_argument('--name',         type=str, default='antiuav_rgbt')
    p.add_argument('--dataset-root', type=str, default=str(DATASET_ROOT),
                   help='Path to Anti-UAV-RGBT root. On HPC pass $TMPDIR '
                        'copy to avoid NFS I/O bottleneck.')
    p.add_argument('--frames-root',  type=str, default=None,
                   help='Path to pre-extracted JPEG frames '
                        '(output of extract_frames_antiuav_rgbt.py). '
                        'If set, skips VideoCapture random seeks entirely.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Override globals from args
    EPOCHS       = args.epochs
    BATCH_SIZE   = args.batch_size
    IMG_SIZE     = args.imgsz
    WORKERS      = args.workers
    DATASET_ROOT = Path(args.dataset_root)
    FRAMES_ROOT  = args.frames_root

    # ── DDP detection ────────────────────────────────────────────────────────
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE automatically.
    # If not set, fall back to single-GPU mode.
    local_rank  = int(os.environ.get('LOCAL_RANK', -1))
    rank        = int(os.environ.get('RANK',       -1))
    world_size  = int(os.environ.get('WORLD_SIZE',  1))

    if local_rank != -1:
        # DDP mode: each process is pinned to one GPU by torchrun
        # Timeout set to 2 h: validation on rank 0 with VideoCapture takes
        # ~40 min; the default 10-min NCCL watchdog kills waiting ranks.
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = select_device(args.device)

    # Only rank 0 creates the save_dir (others reuse the same path)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    if rank in (-1, 0):
        save_dir = Path(increment_path(SAVE_ROOT / args.name, exist_ok=False))
        # Broadcast the chosen path to other ranks so all use the same directory
        if rank == 0:
            path_str = str(save_dir)
    else:
        save_dir = None   # filled in after broadcast below

    # Share the save_dir path across all ranks
    if rank != -1:
        # Pack path into a fixed-length tensor and broadcast
        path_bytes = str(save_dir if rank == 0 else '').encode().ljust(512, b'\x00')
        path_t = torch.ByteTensor(list(path_bytes)).to(device)
        dist.broadcast(path_t, src=0)
        save_dir = Path(bytes(path_t.tolist()).rstrip(b'\x00').decode())

    train(save_dir, device, rank=rank, world_size=world_size)
