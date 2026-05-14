#!/usr/bin/env python3
"""
train_stage2.py
---------------
Stage 2: Teacher-Student UDA — fine-tune YOLOMG on Anti-UAV410 IR.
Supports both single-GPU and 4-GPU DDP via torchrun.

Three-stage continual learning curriculum:
  Stage 1  →  Full supervision on AntiUAVRGBT IR  (establishes T1 mAP ceiling)
  Stage 2  →  Teacher-Student UDA on AntiUAV410   (THIS SCRIPT)
  Stage 3  →  Continual fine-tuning on CST with Scale-Stratified Herding

Method
------
Teacher : frozen Stage 1 model (best.pt from Stage 1, no gradients)
Student : copy of Stage 1 weights, fine-tuned on AntiUAV410

Loss  = L_det  +  λ_kd · L_kd
  L_det : standard YOLO detection loss vs AntiUAV410 GT annotations
  L_kd  : response-level knowledge distillation — MSE between student and
          teacher raw prediction grids (3 scales, before decoding).
          This prevents catastrophic forgetting of T1 while adapting to T2.

Forgetting Measure tracking
---------------------------
At the end of each epoch the EMA student is also evaluated on the
AntiUAVRGBT val split.  The best T1 mAP observed here becomes
mAP_T1_after_T2 for computing:
  FM = mAP_T1_after_T2  −  mAP_T1_after_T1   (negative → forgetting)

DDP notes
---------
  - Student is wrapped in DDP; teacher stays plain (frozen, no gradients)
  - find_unused_parameters=True: zeros img2 skips the motion branch in
    Concat3, leaving backbone1 parameters with no gradient — same issue
    as Stage 1
  - ComputeLoss uses de_parallel(student) to avoid the DDP proxy
    attribute-access issue (model.hyp etc.)
  - Distributed validation: DistributedSampler + all_gather_object for
    both T1 and T2 val sets — prevents NCCL timeout from rank-0-only val
  - persistent_workers=False: prevents VideoCapture worker CPU-RAM leak
    over long jobs (caused Stage 1 OOM at epoch 83)
  - Workers: 4 per GPU (16 total at 4 GPUs) — 8 per process caused OOM
    in Stage 1 (32 workers × VideoCapture buffers > 120 GB RAM)
  - save_dir broadcast via broadcast_object_list (cleaner than byte padding)
  - EMA only on rank 0
  - Loss all_reduce for consistent epoch-level logging
  - Early stopping: computed on rank 0, broadcast to all ranks

Usage:
  # Single GPU
  python train_stage2_ddp.py --weights .../best.pt

  # 4-GPU DDP
  torchrun --standalone --nproc_per_node=4 train_stage2_ddp.py --weights .../best.pt

SLURM:
  sbatch run_stage2_ddp.sh

Outputs:
  /projects/prjs2041/runs/stage2/<name>/
      weights/last.pt
      weights/best.pt
      results.csv
      stage2_t2_mAP.txt   ← best mAP@0.5 on AntiUAV410 val
      stage2_t1_mAP.txt   ← T1 mAP after T2 (for Forgetting Measure)
"""

import sys
import os
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
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT      = Path('/projects/prjs2041/YOLOMG')
UAV_CODE         = Path('/projects/prjs2041/uav_code')
ANTIUAV410_ROOT  = Path('/projects/prjs2041/datasets/Anti-UAV410')
ANTIUAV_RGBT_ROOT= Path('/projects/prjs2041/datasets/Anti-UAV-RGBT')
SAVE_ROOT        = Path('/projects/prjs2041/runs/stage2')
LOGS_DIR         = Path('/projects/prjs2041/logs')

# Default Stage 1 weights (override via --weights argument)
DEFAULT_STAGE1_WEIGHTS = Path('/projects/prjs2041/runs/stage1/antiuav_rgbt14/weights/best.pt')

sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

# ── YOLOMG imports ─────────────────────────────────────────────────────────────
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import (
    colorstr, increment_path, init_seeds,
    non_max_suppression, xywh2xyxy
)
from utils.torch_utils import select_device, ModelEMA, de_parallel, EarlyStopping
from utils.metrics import ap_per_class, box_iou

# ── Our dataset classes ────────────────────────────────────────────────────────
from datasets import AntiUAV410Dataset, AntiUAVRGBTDataset

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Fine-tuning: lower LR than Stage 1 (1e-2 → 1e-3), same regularisation
HYP = {
    'lr0':             1e-3,
    'lrf':             1e-2,   # cosine final ratio
    'momentum':        0.937,
    'weight_decay':    5e-4,
    'warmup_epochs':   1.0,    # shorter warmup for fine-tuning
    'warmup_momentum': 0.8,
    'warmup_bias_lr':  0.01,
    'box':             0.05,
    'cls':             0.5,
    'cls_pw':          1.0,
    'obj':             1.0,
    'obj_pw':          1.0,
    'iou_t':           0.20,
    'anchor_t':        4.0,
    'fl_gamma':        0.0,
    'label_smoothing': 0.0,
    # Augmentation — minimal for IR
    'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.2,
    'degrees': 0.0, 'translate': 0.1, 'scale': 0.3,
    'shear': 0.0, 'perspective': 0.0,
    'flipud': 0.0, 'fliplr': 0.5,
    'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
}

# ── Fixed config ───────────────────────────────────────────────────────────────
IMG_SIZE   = 640
EPOCHS     = 50
BATCH_SIZE = 16        # per GPU
WORKERS    = 4         # per GPU — 8 caused CPU-RAM OOM in Stage 1
SEED       = 42
NC         = 1
NAMES      = ['UAV']
MODEL_CFG  = str(YOLOMG_ROOT / 'models' / 'dual_uav2.yaml')

KD_WEIGHT  = 1.0

CONF_THRES = 0.001
IOU_THRES  = 0.6
MAP_IOU    = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Dataset wrappers
# ══════════════════════════════════════════════════════════════════════════════

def _letterbox_tensor(img: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (3, H, W) float tensor to (3, size, size) with letterboxing."""
    _, h, w = img.shape
    scale   = size / max(h, w)
    nh, nw  = int(round(h * scale)), int(round(w * scale))
    img = F.interpolate(img.unsqueeze(0), size=(nh, nw),
                        mode='bilinear', align_corners=False).squeeze(0)
    pad_h, pad_w = size - nh, size - nw
    top,   left  = pad_h // 2, pad_w // 2
    img = F.pad(img, (left, pad_w - left, top, pad_h - top), value=0.5)
    return img


def _collate(batch):
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
    return imgs, imgs2, targets, list(paths), list(paths), None, None


class Stage2Dataset(Dataset):
    """Anti-UAV410 wrapper for YOLOMG dual-input (img2 = zeros)."""

    def __init__(self, root: Path, split: str, imgsz: int = 640):
        self.ds    = AntiUAV410Dataset(root=root, split=split)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = _letterbox_tensor(sample['image'], self.imgsz)
        img2   = torch.zeros_like(img)
        meta   = sample['meta']
        path   = f"{meta.get('seq','seq')}_{meta.get('frame',idx)}"
        return img, img2, sample['labels'], path

    collate_fn = staticmethod(_collate)


class RGBTValDataset(Dataset):
    """Anti-UAV-RGBT val wrapper — used for FM tracking only."""

    def __init__(self, root: Path, split: str = 'val', imgsz: int = 640):
        self.ds    = AntiUAVRGBTDataset(root=root, split=split)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = _letterbox_tensor(sample['image'], self.imgsz)
        img2   = torch.zeros_like(img)
        meta   = sample['meta']
        path   = f"{meta.get('seq','seq')}_{meta.get('frame',idx)}"
        return img, img2, sample['labels'], path

    collate_fn = staticmethod(_collate)


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Distillation loss
# ══════════════════════════════════════════════════════════════════════════════

def kd_loss(student_preds: list, teacher_preds: list) -> torch.Tensor:
    """
    Response-level KD: MSE between student and teacher raw prediction grids.
    Each GPU computes this locally — DDP averages gradients in the backward
    pass, so there is no need to all-reduce the KD loss explicitly.
    """
    loss = torch.tensor(0.0, device=student_preds[0].device)
    for s, t in zip(student_preds, teacher_preds):
        loss = loss + F.mse_loss(s.float(), t.float().detach())
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_stats(model, loader, device, imgsz=640):
    """
    Run inference on this rank's shard and return raw detection stats.
    Returns list of (correct, conf, pred_cls, target_cls) tuples.
    """
    model.eval()
    stats = []

    for imgs, imgs2, targets, *_ in loader:
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

            if len(det) == 0:
                if nl:
                    stats.append((
                        torch.zeros(0, 1, dtype=torch.bool),
                        torch.zeros(0), torch.zeros(0), tcls
                    ))
                continue

            correct = torch.zeros(len(det), 1, dtype=torch.bool, device=device)
            if nl:
                tbox = xywh2xyxy(gt[:, 1:5]) * imgsz
                iou  = box_iou(tbox.to(device), det[:, :4])
                if iou.numel():
                    x = torch.where(iou >= MAP_IOU)
                    if x[0].shape[0]:
                        matches = torch.cat((
                            torch.stack(x, 1),
                            iou[x[0], x[1]][:, None]
                        ), 1).cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1],
                                                        return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0],
                                                        return_index=True)[1]]
                        correct[matches[:, 1].astype(int), 0] = True
            else:
                correct = torch.zeros(len(det), 1, dtype=torch.bool)

            stats.append((
                correct.cpu(),
                det[:, 4].cpu(),
                det[:, 5].cpu(),
                tcls
            ))

    model.train()
    return stats


def _compute_map(all_stats):
    """Compute mAP@0.5 and mAP@0.5:0.95 from aggregated stats."""
    if not all_stats:
        return 0.0, 0.0
    s = [np.concatenate(x, 0) for x in zip(*all_stats)]
    if len(s) and s[0].any():
        _, _, _, _, _, ap, _ = ap_per_class(*s, plot=False, names={0: 'UAV'})
        map50 = float(ap[:, 0].mean()) if ap.ndim == 2 else float(ap[0])
        mapxx = float(ap.mean())       if ap.ndim == 2 else float(ap.mean())
    else:
        map50, mapxx = 0.0, 0.0
    return map50, mapxx


def evaluate_ddp(model, loader, device, imgsz, rank, world_size):
    """
    Distributed evaluation: each rank collects stats on its shard,
    all_gather_object sends everything to rank 0 for mAP computation.
    Prevents NCCL idle-rank timeout (root cause of Stage 1 job 22501881).
    """
    local_stats = _collect_stats(model, loader, device, imgsz)

    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_stats)
    else:
        gathered = [local_stats]

    if rank in (-1, 0):
        all_stats = []
        for shard in gathered:
            all_stats.extend(shard)
        return _compute_map(all_stats)
    else:
        return 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> Model:
    """
    Build a fresh YOLOMG model and load EMA weights from a Stage 1 checkpoint.
    weights_only=False required for YOLOMG checkpoint format.
    """
    model = Model(MODEL_CFG, ch=3, ch2=3, nc=NC).to(device)
    model.nc    = NC
    model.hyp   = HYP
    model.names = NAMES

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if 'ema' in ckpt and ckpt['ema'] is not None:
        state_dict = ckpt['ema'].float().state_dict()
        src = 'EMA'
    else:
        state_dict = ckpt['model'].float().state_dict()
        src = 'model'

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[WARN] {len(missing)} missing keys from {src}: {missing[:5]}')
    if unexpected:
        print(f'[WARN] {len(unexpected)} unexpected keys from {src}')

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(save_dir: Path, device: torch.device, stage1_weights: Path,
          rank: int = -1, local_rank: int = -1, world_size: int = 1):

    is_main    = rank in (-1, 0)
    is_ddp     = world_size > 1

    # Different seed per rank → different augmentation per GPU
    init_seeds(SEED + max(rank, 0))

    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(exist_ok=True)

    # Barrier: all ranks wait until rank 0 has created save_dir
    if is_ddp:
        dist.barrier()

    # ── Logging (rank 0 only to avoid duplicate lines) ───────────────────────
    log = logging.getLogger()
    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s  %(message)s',
            handlers=[
                logging.FileHandler(save_dir / 'train.log'),
                logging.StreamHandler(sys.stdout),
            ]
        )
        log.info(f'Stage 2 Teacher-Student UDA  →  {save_dir}')
        log.info(f'Stage 1 weights: {stage1_weights}')
        log.info(f'Device: {device}  |  IMG_SIZE={IMG_SIZE}  EPOCHS={EPOCHS}  '
                 f'BS={BATCH_SIZE}  world_size={world_size}')
        log.info(f'KD weight λ = {KD_WEIGHT}')

        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(HYP, f)

    # ── Datasets ──────────────────────────────────────────────────────────────
    if is_main:
        log.info('Loading datasets …')

    train_ds   = Stage2Dataset(ANTIUAV410_ROOT,    split='train', imgsz=IMG_SIZE)
    val_ds_t2  = Stage2Dataset(ANTIUAV410_ROOT,    split='val',   imgsz=IMG_SIZE)
    val_ds_t1  = RGBTValDataset(ANTIUAV_RGBT_ROOT, split='val',   imgsz=IMG_SIZE)

    if is_main:
        log.info(f'  T2 train : {len(train_ds):,} frames   (AntiUAV410)')
        log.info(f'  T2 val   : {len(val_ds_t2):,} frames   (AntiUAV410)')
        log.info(f'  T1 val   : {len(val_ds_t1):,} frames   (AntiUAVRGBT — FM tracking)')

    # Training: DistributedSampler partitions data across ranks
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=max(rank, 0), shuffle=True) \
                    if is_ddp else None

    # Val: DistributedSampler so all ranks participate — prevents NCCL timeout
    val_sampler_t2 = DistributedSampler(val_ds_t2, num_replicas=world_size,
                                         rank=max(rank, 0), shuffle=False) \
                     if is_ddp else None
    val_sampler_t1 = DistributedSampler(val_ds_t1, num_replicas=world_size,
                                         rank=max(rank, 0), shuffle=False) \
                     if is_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=WORKERS, pin_memory=True,
        persistent_workers=False,          # prevents VideoCapture RAM leak
        collate_fn=Stage2Dataset.collate_fn,
    )
    val_loader_t2 = DataLoader(
        val_ds_t2, batch_size=BATCH_SIZE * 2,
        sampler=val_sampler_t2, shuffle=False,
        num_workers=WORKERS, pin_memory=True,
        persistent_workers=False,
        collate_fn=Stage2Dataset.collate_fn,
    )
    val_loader_t1 = DataLoader(
        val_ds_t1, batch_size=BATCH_SIZE * 2,
        sampler=val_sampler_t1, shuffle=False,
        num_workers=WORKERS, pin_memory=True,
        persistent_workers=False,
        collate_fn=RGBTValDataset.collate_fn,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    if is_main:
        log.info(f'Loading student from {stage1_weights} …')
    student = load_model_from_ckpt(stage1_weights, device)

    if is_main:
        log.info('Creating frozen teacher (same weights) …')
    teacher = load_model_from_ckpt(stage1_weights, device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # EMA on rank 0 only (uses de_parallel to get raw module)
    ema = ModelEMA(student) if is_main else None

    # Wrap student in DDP — AFTER setting .nc/.hyp/.names so de_parallel works
    # find_unused_parameters=True: zeros img2 skips motion branch (backbone1)
    if is_ddp:
        student = DDP(student, device_ids=[local_rank],
                      find_unused_parameters=True)

    # ComputeLoss MUST use de_parallel to avoid DDP proxy attribute error
    compute_loss = ComputeLoss(de_parallel(student))

    if is_main:
        log.info(f'  Parameters: {sum(p.numel() for p in de_parallel(student).parameters()):,}')
        log.info(f'  Trainable:  {sum(p.numel() for p in de_parallel(student).parameters() if p.requires_grad):,}')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    raw = de_parallel(student)
    g0, g1, g2 = [], [], []
    for v in raw.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    optimizer = SGD(g0, lr=HYP['lr0'], momentum=HYP['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': HYP['weight_decay']})
    optimizer.add_param_group({'params': g2})
    del g0, g1, g2, raw

    lf = lambda x: ((1 - math.cos(x * math.pi / EPOCHS)) / 2) * (HYP['lrf'] - 1) + 1
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    # ── AMP + early stopping ──────────────────────────────────────────────────
    scaler  = amp.GradScaler(enabled=(device.type != 'cpu'))
    stopper = EarlyStopping(patience=15) if is_main else None

    # ── Results CSV (rank 0 only) ─────────────────────────────────────────────
    csv_path  = save_dir / 'results.csv'
    last_path = save_dir / 'weights' / 'last.pt'
    best_path = save_dir / 'weights' / 'best.pt'

    if is_main:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'loss_box', 'loss_obj', 'loss_cls', 'loss_kd',
                'loss_total', 'mAP50_T2', 'mAP50_T1', 'lr'
            ])

    # ── Training loop ─────────────────────────────────────────────────────────
    nb       = len(train_loader)
    nw       = max(round(HYP['warmup_epochs'] * nb), 50)
    best_t2  = 0.0
    best_t1  = 0.0

    if is_main:
        log.info(f'\nStarting Stage 2 training for {EPOCHS} epochs …')
        log.info(f'  L = L_det + {KD_WEIGHT} × L_kd\n')

    for epoch in range(EPOCHS):

        # DistributedSampler must be told the epoch for correct shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        student.train()
        mloss = torch.zeros(4, device=device)   # box, obj, cls, kd

        for i, (imgs, imgs2, targets, *_) in enumerate(train_loader):
            ni    = i + nb * epoch
            imgs  = imgs.to(device,  non_blocking=True)
            imgs2 = imgs2.to(device, non_blocking=True)
            targets = targets.to(device)

            # Warmup LR/momentum
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

            with amp.autocast(enabled=(device.type != 'cpu')):
                # Student forward (train mode → raw prediction grids)
                student_preds = student(imgs, imgs2)
                det_loss, loss_items = compute_loss(student_preds, targets)

                # Teacher forward (eval, no_grad, runs independently on each GPU)
                with torch.no_grad():
                    teacher_out   = teacher(imgs, imgs2)
                    teacher_grids = teacher_out[1] \
                                    if isinstance(teacher_out, tuple) else teacher_out

                # KD loss
                if isinstance(student_preds, (list, tuple)) and \
                   isinstance(teacher_grids, (list, tuple)):
                    kd = kd_loss(list(student_preds), list(teacher_grids))
                else:
                    kd = torch.tensor(0.0, device=device)

                total_loss = det_loss + KD_WEIGHT * kd

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA update on rank 0 only (de_parallel unwraps DDP)
            if ema is not None:
                ema.update(de_parallel(student))

            # Running mean loss
            batch_items = torch.cat([loss_items, kd.unsqueeze(0).detach()])
            mloss = (mloss * i + batch_items) / (i + 1)

            # Step logging (rank 0 only)
            if is_main and i % 50 == 0:
                mem = f'{torch.cuda.memory_reserved() / 1e9:.2f}G' \
                      if torch.cuda.is_available() else 'cpu'
                log.info(
                    f'  [{epoch:3d}/{EPOCHS-1}][{i:4d}/{nb}]  '
                    f'mem={mem}  '
                    f'box={mloss[0]:.4f}  obj={mloss[1]:.4f}  '
                    f'cls={mloss[2]:.4f}  kd={mloss[3]:.4f}'
                )

        scheduler.step()

        # ── All-reduce loss for consistent epoch logging ───────────────────
        if is_ddp:
            dist.all_reduce(mloss, op=dist.ReduceOp.SUM)
            mloss /= world_size

        # ── End-of-epoch evaluation (distributed) ────────────────────────
        t_epoch  = datetime.now().strftime('%H:%M:%S') if is_main else ''
        eval_model = ema.ema if ema is not None else de_parallel(student)

        map50_t2, _ = evaluate_ddp(eval_model, val_loader_t2,
                                    device, IMG_SIZE, rank, world_size)
        map50_t1, _ = evaluate_ddp(eval_model, val_loader_t1,
                                    device, IMG_SIZE, rank, world_size)

        lr_now         = optimizer.param_groups[0]['lr']
        total_loss_val = mloss.sum().item()

        # ── Rank-0 reporting, CSV, checkpoint ────────────────────────────
        if is_main:
            best_t1 = max(best_t1, map50_t1)

            log.info(
                f'\nEpoch {epoch:3d}/{EPOCHS-1}  [{t_epoch}]  '
                f'loss={total_loss_val:.4f}  '
                f'mAP@0.5(T2/UAV410)={map50_t2:.4f}  '
                f'mAP@0.5(T1/RGBT)={map50_t1:.4f}\n'
            )

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch,
                    mloss[0].item(), mloss[1].item(),
                    mloss[2].item(), mloss[3].item(),
                    total_loss_val, map50_t2, map50_t1, lr_now
                ])

            ckpt = {
                'epoch':     epoch,
                'best_t2':   best_t2,
                'best_t1':   best_t1,
                'model':     deepcopy(de_parallel(student)).half(),
                'ema':       deepcopy(ema.ema).half(),
                'updates':   ema.updates,
                'optimizer': optimizer.state_dict(),
                'hyp':       HYP,
                'kd_weight': KD_WEIGHT,
                'date':      datetime.now().isoformat(),
            }
            torch.save(ckpt, last_path)
            if map50_t2 > best_t2:
                best_t2 = map50_t2
                torch.save(ckpt, best_path)
                log.info(f'  ✓ New best T2 mAP@0.5 = {best_t2:.4f}  (saved {best_path})')

        # ── Early stopping: rank 0 decides, all ranks obey ───────────────
        stop = torch.zeros(1, device=device)
        if is_main and stopper(epoch=epoch, fitness=map50_t2):
            log.info('Early stopping triggered.')
            stop.fill_(1)
        if is_ddp:
            dist.broadcast(stop, src=0)
        if stop.item():
            break

    # ── Final report (rank 0 only) ────────────────────────────────────────────
    if is_main:
        log.info(f'\n{"=" * 60}')
        log.info(f'Stage 2 complete.')
        log.info(f'Best T2 mAP@0.5 (AntiUAV410)    = {best_t2:.4f}')
        log.info(f'Best T1 mAP@0.5 (AntiUAVRGBT)   = {best_t1:.4f}  ← FM numerator')
        log.info(f'Weights: {best_path}')
        log.info(f'Results: {csv_path}')

        t2_txt = save_dir / 'stage2_t2_mAP.txt'
        t2_txt.write_text(f'{best_t2:.6f}\n')
        log.info(f'T2 mAP written to {t2_txt}')

        t1_txt = save_dir / 'stage2_t1_mAP.txt'
        t1_txt.write_text(f'{best_t1:.6f}\n')
        log.info(f'T1 mAP after T2 written to {t1_txt}  ← FM = this − stage1_mAP.txt')

    if is_ddp:
        dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Stage 2: Teacher-Student UDA on AntiUAV410')
    p.add_argument('--weights',    type=str, default=str(DEFAULT_STAGE1_WEIGHTS))
    p.add_argument('--epochs',     type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--imgsz',      type=int, default=IMG_SIZE)
    p.add_argument('--workers',    type=int, default=WORKERS)
    p.add_argument('--device',     type=str, default='')
    p.add_argument('--kd-weight',  type=float, default=KD_WEIGHT)
    p.add_argument('--name',       type=str, default='antiuav410')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    IMG_SIZE   = args.imgsz
    WORKERS    = args.workers
    KD_WEIGHT  = args.kd_weight

    # ── DDP setup ─────────────────────────────────────────────────────────────
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
    RANK       = int(os.environ.get('RANK',       -1))
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE',  1))

    if LOCAL_RANK != -1:
        # Launched via torchrun
        assert torch.cuda.device_count() > LOCAL_RANK, \
            f'Not enough GPUs: need {LOCAL_RANK+1}, have {torch.cuda.device_count()}'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=2)   # prevents NCCL watchdog timeout during long val
        )
    else:
        device = select_device(args.device)
        RANK = -1

    # ── Save dir: create on rank 0, broadcast to all ──────────────────────────
    if RANK in (-1, 0):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        save_dir = Path(increment_path(SAVE_ROOT / args.name, exist_ok=False))
    else:
        save_dir = None

    if WORLD_SIZE > 1:
        # broadcast_object_list is cleaner than the byte-padding approach
        obj = [str(save_dir)]
        dist.broadcast_object_list(obj, src=0)
        save_dir = Path(obj[0])

    train(save_dir, device, stage1_weights=Path(args.weights),
          rank=RANK, local_rank=LOCAL_RANK, world_size=WORLD_SIZE)
