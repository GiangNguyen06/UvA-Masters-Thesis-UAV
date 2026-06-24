#!/usr/bin/env python3
"""
train_stage3.py
---------------
Stage 3: Continual fine-tuning on CST Anti-UAV with optional replay.

Three experimental conditions (controlled by --replay-mode):

  Naive baseline  (--replay-mode none):
    Fine-tune Stage 2 best.pt on CST with detection loss only.
    No anti-forgetting mechanism. Establishes FM_naive.

  Random Stratified (--replay-mode random_stratified --replay-buffer <path>):
    Fine-tune on CST + replay buffer sampled equally across size strata
    but selected RANDOMLY within each stratum (no herding).
    Isolates the contribution of strata balance vs. herding selection.
    Establishes FM_random.

  Scale-Stratified Herding (--replay-mode herding --replay-buffer <path>):
    Same as random_stratified, but exemplars selected via greedy herding
    in feature space (iCaRL-style) within each stratum.
    This is the thesis contribution. Establishes FM_herding.

Thesis claim: FM_herding > FM_random > FM_naive  (less forgetting).

Forgetting Measures (two tracked simultaneously):
  fm_abs    = mAP_T1_after_T3 − mAP_T1_after_T1  (T1 baseline = 0.6725)
  fm_stage3 = mAP_T1_after_T3 − mAP_T1_after_T2  (Stage 2 retention as reference)

  fm_abs enables cross-stage comparison.
  fm_stage3 is the honest Stage-3-specific forgetting measure.

  Both are negative when forgetting occurs; less negative = less forgetting.
  Both are evaluated AT THE EPOCH WHERE BEST T3 mAP IS ACHIEVED.

Usage:
  # Naive baseline (4-GPU DDP)
  torchrun --standalone --nproc_per_node=4 train_stage3.py \\
      --weights  /projects/prjs2041/runs/stage2/seed42/weights/best.pt \\
      --replay-mode none --name naive

  # Random stratified
  torchrun --standalone --nproc_per_node=4 train_stage3.py \\
      --weights      /projects/prjs2041/runs/stage2/seed42/weights/best.pt \\
      --replay-mode  random_stratified \\
      --replay-buffer /projects/prjs2041/runs/stage1/antiuav_rgbt15/herding_buffer_random.pt \\
      --name random_stratified

  # Scale-Stratified Herding
  torchrun --standalone --nproc_per_node=4 train_stage3.py \\
      --weights      /projects/prjs2041/runs/stage2/seed42/weights/best.pt \\
      --replay-mode  herding \\
      --replay-buffer /projects/prjs2041/runs/stage1/antiuav_rgbt15/herding_buffer.pt \\
      --name herding

Outputs:
  /projects/prjs2041/runs/stage3/<name>/
      weights/best.pt
      weights/last.pt
      results.csv        — per-epoch metrics (see CSV header below)
      stage3_mAP.txt     — best T3 mAP (CST val)
      stage3_fm.txt      — fm_abs and fm_stage3 at best T3 epoch
      stage3_t2_baseline.txt  — mAP_T1_after_T2 (pre-Stage3 T1 level)
      stage3_condition.txt    — condition name
"""

import sys
import os
import csv
import math
import argparse
import logging
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
YOLOMG_ROOT = Path('/projects/prjs2041/YOLOMG')
UAV_CODE    = Path('/projects/prjs2041/uav_code')
CST_ROOT    = Path('/projects/prjs2041/datasets/CST-AntiUAV')
RGBT_ROOT   = Path('/projects/prjs2041/datasets/Anti-UAV-RGBT')
SAVE_ROOT   = Path('/projects/prjs2041/runs/stage3')
LOGS_DIR    = Path('/projects/prjs2041/logs')

DEFAULT_STAGE2_WEIGHTS = Path(
    '/projects/prjs2041/runs/stage2/seed42/weights/best.pt')

sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import increment_path, init_seeds, non_max_suppression, xywh2xyxy
from utils.torch_utils import select_device, ModelEMA, de_parallel, EarlyStopping
from utils.metrics import ap_per_class, box_iou

from datasets import CSTDataset, AntiUAVRGBTDataset

# ── Hyperparameters ────────────────────────────────────────────────────────────
HYP = {
    'lr0':             5e-4,    # lower LR for Stage 3 (stage 2 used 1e-3)
    'lrf':             1e-2,
    'momentum':        0.937,
    'weight_decay':    5e-4,
    'warmup_epochs':   1.0,
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
    'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.2,
    'degrees': 0.0, 'translate': 0.1, 'scale': 0.3,
    'shear': 0.0, 'perspective': 0.0,
    'flipud': 0.0, 'fliplr': 0.5,
    'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
}

IMG_SIZE      = 640
EPOCHS        = 50
BATCH_SIZE    = 16       # per GPU
WORKERS       = 4        # per GPU
SEED          = 42
NC            = 1
NAMES         = ['UAV']
MODEL_CFG     = str(YOLOMG_ROOT / 'models' / 'dual_uav2.yaml')
REPLAY_RATIO  = 0.25     # fraction of each batch filled with replay exemplars
                         # 0.25 × 16 = 4 replay samples per CST batch of 16
REPLAY_WEIGHT = 4.0      # scalar applied to replay loss to compensate for
                         # smaller replay batch (4 vs 16 CST samples)
T1_BASELINE   = 0.6725   # mAP_T1_after_T1 from antiuav_rgbt15 (epoch 49)

CONF_THRES    = 0.001
IOU_THRES     = 0.6

# ── UAV-specific size bins (matching eval_full_analysis.py) ───────────────────
# All Anti-UAV-RGBT GT boxes have side < 96px, so COCO bins are useless here.
SIZE_BINS_T1 = [
    ('tiny',   0,      16**2),    # area < 256 px²   (side < 16px)
    ('small',  16**2,  32**2),    # area 256–1024 px² (side 16–32px)
    ('normal', 32**2,  64**2),    # area 1024–4096 px² (side 32–64px)
    ('large',  64**2,  float('inf')),  # area > 4096 px²   (side > 64px)
]


def _size_cat(w_px: float, h_px: float) -> str:
    area = w_px * h_px
    for name, lo, hi in SIZE_BINS_T1:
        if lo <= area < hi:
            return name
    return 'large'


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


class Stage3Dataset(Dataset):
    """CST Anti-UAV wrapper for YOLOMG (img2 = zeros)."""

    def __init__(self, root: Path, split: str, imgsz: int = 640):
        self.ds    = CSTDataset(root=root, split=split, skip_empty=False)
        self.imgsz = imgsz

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = _letterbox_tensor(sample['image'], self.imgsz)
        img2   = torch.zeros_like(img)
        return img, img2, sample['labels'], sample.get('meta', {})


class RGBTValDataset(Dataset):
    """Anti-UAV-RGBT val wrapper for T1 retention tracking."""

    def __init__(self, root: Path, imgsz: int = 640):
        self.ds    = AntiUAVRGBTDataset(root=root, split='val')
        self.imgsz = imgsz

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = _letterbox_tensor(sample['image'], self.imgsz)
        img2   = torch.zeros_like(img)
        return img, img2, sample['labels'], sample.get('meta', {})


class ReplayDataset(Dataset):
    """
    Wraps a replay buffer .pt file as a PyTorch Dataset.
    Works for both herding and random-stratified buffers.
    Images are loaded on-the-fly from paths stored in metas.
    """

    def __init__(self, buffer_path: Path, imgsz: int = 640):
        import cv2
        self.cv2   = cv2
        self.imgsz = imgsz
        buf        = torch.load(buffer_path, map_location='cpu', weights_only=False)
        self.metas  = buf['metas']
        self.labels = buf['labels']
        self.strata = buf['strata']
        print(f'  Replay buffer: {len(self.metas)} exemplars loaded from {buffer_path}')

    def __len__(self): return len(self.metas)

    def __getitem__(self, idx):
        meta   = self.metas[idx]
        labels = self.labels[idx]

        # Meta from AntiUAVRGBTDataset contains 'seq_dir' and 'frame' (int),
        # not 'img_path'. Reconstruct the IR frame from the video file.
        img    = None
        seq_dir   = meta.get('seq_dir')
        frame_idx = meta.get('frame')

        if seq_dir and frame_idx is not None:
            video_path = Path(seq_dir) / 'infrared.mp4'
            if video_path.exists():
                cap = self.cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        if img is None:
            img = torch.zeros(3, self.imgsz, self.imgsz)

        img  = _letterbox_tensor(img, self.imgsz)
        img2 = torch.zeros_like(img)

        return img, img2, labels, meta


def _collate(batch):
    imgs, imgs2, labels_list, metas = zip(*batch)
    imgs  = torch.stack(imgs)
    imgs2 = torch.stack(imgs2)
    targets = []
    for i, lbl in enumerate(labels_list):
        if len(lbl) > 0:
            t  = torch.as_tensor(lbl, dtype=torch.float32)
            bi = torch.full((len(t), 1), float(i))
            targets.append(torch.cat([bi, t], 1))
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6))
    return imgs, imgs2, targets, list(metas)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation — aggregate
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_stats(model, loader, device, imgsz):
    """Collect detection stats for mAP computation. Returns stats list."""
    model.eval()
    stats = []
    for imgs, imgs2, targets, _ in loader:
        imgs    = imgs.to(device)
        imgs2   = imgs2.to(device)
        targets = targets.to(device)
        preds   = model(imgs, imgs2)
        if isinstance(preds, tuple): preds = preds[0]
        preds   = non_max_suppression(preds, CONF_THRES, IOU_THRES)
        for si, det in enumerate(preds):
            gt   = targets[targets[:, 0] == si, 1:]
            nl   = len(gt)
            tcls = gt[:, 0].long().tolist() if nl else []
            if len(det) == 0:
                if nl:
                    stats.append((torch.zeros(0, 1, dtype=torch.bool),
                                  torch.zeros(0), torch.zeros(0), tcls))
                continue
            correct = torch.zeros(len(det), 1, dtype=torch.bool, device=device)
            if nl:
                tbox = xywh2xyxy(gt[:, 1:5]) * imgsz
                iou  = box_iou(tbox, det[:, :4])
                x    = torch.where(iou >= 0.5)
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1).float(),
                                         iou[x[0], x[1]].unsqueeze(1)), 1).cpu().numpy()
                    if matches.shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), 0] = True
            stats.append((correct.cpu(), det[:, 4].cpu(), det[:, 5].cpu(), tcls))
    model.train()
    return stats


def _compute_metrics(all_stats):
    if not all_stats:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    s = [np.concatenate(x, 0) for x in zip(*all_stats)]
    if len(s) and s[0].any():
        _, _, p, r, f1, ap, _ = ap_per_class(*s, plot=False, names={0: 'UAV'})
        map50 = float(ap[:, 0].mean()) if ap.ndim == 2 else float(ap[0])
        mapxx = float(ap.mean())       if ap.ndim == 2 else float(ap.mean())
        return map50, mapxx, float(np.mean(p)), float(np.mean(r)), float(np.mean(f1))
    return 0.0, 0.0, 0.0, 0.0, 0.0


def evaluate_ddp(model, loader, device, imgsz, rank, world_size):
    local_stats = _collect_stats(model, loader, device, imgsz)
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_stats)
    else:
        gathered = [local_stats]
    if rank in (-1, 0):
        all_stats = []
        for shard in gathered: all_stats.extend(shard)
        return _compute_metrics(all_stats)
    return 0.0, 0.0, 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation — per-stratum (T1 only)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_stats_with_sizes(model, loader, device, imgsz):
    """
    Like _collect_stats but also returns gt_sizes: list of stratum names,
    one per GT object in the order they were encountered.
    Used for per-stratum mAP computation.
    """
    model.eval()
    stats    = []
    gt_sizes = []
    for imgs, imgs2, targets, _ in loader:
        imgs    = imgs.to(device)
        imgs2   = imgs2.to(device)
        targets = targets.to(device)
        preds   = model(imgs, imgs2)
        if isinstance(preds, tuple): preds = preds[0]
        preds   = non_max_suppression(preds, CONF_THRES, IOU_THRES)
        for si, det in enumerate(preds):
            gt   = targets[targets[:, 0] == si, 1:]
            nl   = len(gt)
            tcls = gt[:, 0].long().tolist() if nl else []
            if nl:
                for row in gt[:, 1:5].cpu().numpy():
                    _, _, wn, hn = row
                    gt_sizes.append(_size_cat(wn * imgsz, hn * imgsz))
            if len(det) == 0:
                if nl:
                    stats.append((torch.zeros(0, 1, dtype=torch.bool),
                                  torch.zeros(0), torch.zeros(0), tcls))
                continue
            correct = torch.zeros(len(det), 1, dtype=torch.bool, device=device)
            if nl:
                tbox = xywh2xyxy(gt[:, 1:5]) * imgsz
                iou  = box_iou(tbox, det[:, :4])
                x    = torch.where(iou >= 0.5)
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1).float(),
                                         iou[x[0], x[1]].unsqueeze(1)), 1).cpu().numpy()
                    if matches.shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), 0] = True
            stats.append((correct.cpu(), det[:, 4].cpu(), det[:, 5].cpu(), tcls))
    model.train()
    return stats, gt_sizes


def _stratum_map(all_stats, gt_sizes):
    """
    Compute mAP50 per size stratum.
    Mirrors eval_full_analysis.scale_stratified_map().
    Returns dict: stratum_name -> float (mAP50, or nan if no GT in stratum).
    """
    results = {}
    for cat_name, _, _ in SIZE_BINS_T1:
        cat_stats = []
        gt_ptr    = 0
        for correct, conf, pred_cls, tcls in all_stats:
            n_gt = len(tcls)
            if n_gt == 0:
                cat_stats.append((correct, conf, pred_cls, tcls))
            else:
                img_cats = gt_sizes[gt_ptr:gt_ptr + n_gt]
                if cat_name in img_cats:
                    cat_stats.append((correct, conf, pred_cls, tcls))
                gt_ptr += n_gt
        results[cat_name] = _compute_metrics(cat_stats)[0] if cat_stats else float('nan')
    return results


def evaluate_by_stratum_ddp(model, loader, device, imgsz, rank, world_size):
    """
    DDP-aware per-stratum T1 evaluation.
    Returns dict stratum->mAP50 on rank 0; nan dict on other ranks.
    """
    local_stats, local_sizes = _collect_stats_with_sizes(model, loader, device, imgsz)
    if world_size > 1:
        g_stats = [None] * world_size
        g_sizes = [None] * world_size
        dist.all_gather_object(g_stats, local_stats)
        dist.all_gather_object(g_sizes, local_sizes)
    else:
        g_stats = [local_stats]
        g_sizes = [local_sizes]
    if rank in (-1, 0):
        all_stats, all_sizes = [], []
        for s, g in zip(g_stats, g_sizes):
            all_stats.extend(s)
            all_sizes.extend(g)
        return _stratum_map(all_stats, all_sizes)
    return {cat: float('nan') for cat, *_ in SIZE_BINS_T1}


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(save_dir: Path, device, stage2_weights: Path,
          replay_buffer_path: Path = None,
          replay_mode: str = 'none',
          replay_weight: float = REPLAY_WEIGHT,
          save_epoch_ckpts: bool = False, save_epoch_interval: int = 5,
          rank: int = -1, local_rank: int = -1, world_size: int = 1):

    is_main   = rank in (-1, 0)
    is_ddp    = world_size > 1
    init_seeds(SEED + rank if rank != -1 else SEED)
    use_replay = replay_mode != 'none'
    condition  = replay_mode   # 'naive' / 'random_stratified' / 'herding'
    if not use_replay:
        condition = 'naive'

    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(exist_ok=True)

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.handlers = []
    if is_main:
        log.addHandler(logging.FileHandler(save_dir / 'train.log'))
    log.addHandler(logging.StreamHandler())

    if is_main:
        log.info(f'Stage 3 — {condition.upper()} condition  →  {save_dir}')
        log.info(f'Stage 2 weights : {stage2_weights}')
        if use_replay:
            log.info(f'Replay buffer   : {replay_buffer_path}')
            log.info(f'Replay ratio    : {REPLAY_RATIO:.0%} of each batch  '
                     f'({max(1, int(BATCH_SIZE * REPLAY_RATIO))} samples)')
            log.info(f'Replay weight   : {replay_weight}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    if is_main: log.info('\nLoading datasets …')

    cst_train = Stage3Dataset(CST_ROOT,  split='train', imgsz=IMG_SIZE)
    cst_val   = Stage3Dataset(CST_ROOT,  split='val',   imgsz=IMG_SIZE)
    rgbt_val  = RGBTValDataset(RGBT_ROOT, imgsz=IMG_SIZE)

    if is_main:
        log.info(f'  CST train: {len(cst_train):,}  val: {len(cst_val):,}')
        log.info(f'  T1 val (RGBT): {len(rgbt_val):,}')

    if use_replay:
        replay_ds = ReplayDataset(replay_buffer_path, imgsz=IMG_SIZE)
        if is_main: log.info(f'  Replay: {len(replay_ds):,} exemplars')

    train_sampler  = (DistributedSampler(cst_train, num_replicas=world_size,
                                         rank=rank, shuffle=True)
                      if is_ddp else None)
    val_sampler_t3 = (DistributedSampler(cst_val, num_replicas=world_size,
                                          rank=rank, shuffle=False)
                      if is_ddp else None)
    val_sampler_t1 = (DistributedSampler(rgbt_val, num_replicas=world_size,
                                          rank=rank, shuffle=False)
                      if is_ddp else None)

    train_loader  = DataLoader(cst_train, batch_size=BATCH_SIZE,
                               shuffle=(train_sampler is None),
                               sampler=train_sampler, num_workers=WORKERS,
                               pin_memory=True, persistent_workers=False,
                               collate_fn=_collate)
    val_loader_t3 = DataLoader(cst_val, batch_size=BATCH_SIZE * 2, shuffle=False,
                                sampler=val_sampler_t3, num_workers=WORKERS,
                                pin_memory=True, persistent_workers=False,
                                collate_fn=_collate)
    val_loader_t1 = DataLoader(rgbt_val, batch_size=BATCH_SIZE * 2, shuffle=False,
                                sampler=val_sampler_t1, num_workers=WORKERS,
                                pin_memory=True, persistent_workers=False,
                                collate_fn=_collate)

    replay_loader = None
    replay_iter   = None
    if use_replay:
        replay_batch = max(1, int(BATCH_SIZE * REPLAY_RATIO))
        replay_loader = DataLoader(replay_ds, batch_size=replay_batch,
                                   shuffle=True, num_workers=2,
                                   pin_memory=True, persistent_workers=False,
                                   collate_fn=_collate, drop_last=True)
        replay_iter = iter(replay_loader)
        if is_main:
            log.info(f'  Replay batch size: {replay_batch} per step')

    # ── Model ─────────────────────────────────────────────────────────────────
    if is_main: log.info(f'\nLoading Stage 2 weights from {stage2_weights} …')
    ckpt  = torch.load(str(stage2_weights), map_location=device, weights_only=False)
    model = Model(MODEL_CFG, ch=3, ch2=3, nc=NC).to(device)
    model.nc = NC; model.hyp = HYP; model.names = NAMES

    state_src = ckpt.get('ema') or ckpt.get('model')
    sd = state_src.float().state_dict() if state_src else {}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if is_main and missing:
        log.warning(f'  {len(missing)} missing keys')

    if is_ddp:
        gpu_id = local_rank if local_rank != -1 else rank
        model  = DDP(model, device_ids=[gpu_id], output_device=gpu_id,
                     find_unused_parameters=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):        g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):                                   g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  g1.append(v.weight)

    optimizer = SGD(g0, lr=HYP['lr0'], momentum=HYP['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': HYP['weight_decay']})
    optimizer.add_param_group({'params': g2})

    lf        = lambda x: ((1 - math.cos(x * math.pi / EPOCHS)) / 2) * (HYP['lrf'] - 1) + 1
    scheduler = LambdaLR(optimizer, lr_lambda=lf)
    scaler    = amp.GradScaler(enabled=(device.type != 'cpu'))
    ema       = ModelEMA(model) if is_main else None
    stopper   = EarlyStopping(patience=15) if is_main else None

    compute_loss = ComputeLoss(de_parallel(model))

    # ── Measure Stage 2 T1 retention BEFORE any Stage 3 gradient ─────────────
    # This is the honest Stage-3-specific FM reference point.
    if is_main: log.info('\nMeasuring Stage 2 T1 baseline (no gradient steps yet) …')
    eval_model_pre = ema.ema if ema is not None else de_parallel(model)
    map50_t2_baseline, *_ = evaluate_ddp(
        eval_model_pre, val_loader_t1, device, IMG_SIZE, rank, world_size)
    if is_main:
        log.info(f'  mAP_T1_after_T2  = {map50_t2_baseline:.4f}')
        log.info(f'  mAP_T1_after_T1  = {T1_BASELINE:.4f}  (fixed ceiling)')
        log.info(f'  Stage 2 forgetting = {map50_t2_baseline - T1_BASELINE:.4f}\n')
        (save_dir / 'stage3_t2_baseline.txt').write_text(f'{map50_t2_baseline:.6f}\n')

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path  = save_dir / 'results.csv'
    last_path = save_dir / 'weights' / 'last.pt'
    best_path = save_dir / 'weights' / 'best.pt'

    if is_main:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'loss_cst', 'loss_replay', 'loss_total',
                'mAP50_T3', 'P_T3', 'R_T3', 'F1_T3',
                'mAP50_T1', 'P_T1', 'R_T1', 'F1_T1',
                'fm_abs', 'fm_stage3',
                'mAP50_T1_tiny', 'mAP50_T1_small', 'mAP50_T1_normal', 'mAP50_T1_large',
                'lr',
            ])

    # ── Training loop ─────────────────────────────────────────────────────────
    nb               = len(train_loader)
    nw               = max(round(HYP['warmup_epochs'] * nb), 50)
    best_t3          = 0.0
    best_t1          = 0.0
    fm_at_best_t3    = float('nan')   # fm_abs at the epoch best T3 was achieved
    fm_s3_at_best_t3 = float('nan')   # fm_stage3 at the same epoch

    if is_main:
        log.info(f'Starting Stage 3 ({condition}) for {EPOCHS} epochs …\n')

    for epoch in range(EPOCHS):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        mloss_cst    = torch.zeros(3, device=device)
        mloss_replay = torch.zeros(1, device=device)

        for i, (imgs, imgs2, targets, _) in enumerate(train_loader):
            ni      = i + nb * epoch
            imgs    = imgs.to(device, non_blocking=True)
            imgs2   = imgs2.to(device, non_blocking=True)
            targets = targets.to(device)

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi,
                        [HYP['warmup_bias_lr'] if j == 2 else 0.0,
                         x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [HYP['warmup_momentum'], HYP['momentum']])

            optimizer.zero_grad()

            # ── CST forward + backward ────────────────────────────────────────
            # Two separate backward passes are required because YOLOMG contains
            # in-place operations. Combining two forward passes into one backward
            # causes: "variable needed for gradient computation has been modified
            # by an inplace operation". Separate passes accumulate gradients
            # correctly without this conflict.
            #
            # In DDP mode we additionally wrap the FIRST backward in no_sync()
            # so that AllReduce is deferred to the final backward. Without this,
            # DDP fires AllReduce after the first backward, which modifies
            # gradient buckets in-place and increments version counters on
            # tensors still needed for the second backward — causing the same
            # "inplace operation" RuntimeError even with separate passes.
            with amp.autocast(enabled=(device.type != 'cpu')):
                pred_cst = model(imgs, imgs2)
                det_loss_cst, loss_items_cst = compute_loss(pred_cst, targets)

            if is_ddp and use_replay:
                with model.no_sync():
                    scaler.scale(det_loss_cst).backward()
            else:
                scaler.scale(det_loss_cst).backward()

            # ── Replay forward + backward (herding / random_stratified) ───────
            loss_replay = torch.tensor(0.0, device=device)
            if use_replay:
                try:
                    r_imgs, r_imgs2, r_targets, _ = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    r_imgs, r_imgs2, r_targets, _ = next(replay_iter)

                r_imgs    = r_imgs.to(device, non_blocking=True)
                r_imgs2   = r_imgs2.to(device, non_blocking=True)
                r_targets = r_targets.to(device)

                with amp.autocast(enabled=(device.type != 'cpu')):
                    pred_replay = model(r_imgs, r_imgs2)
                    loss_replay, _ = compute_loss(pred_replay, r_targets)

                # This backward triggers DDP AllReduce (syncs both CST + replay grads)
                scaler.scale(replay_weight * loss_replay).backward()

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(de_parallel(model))

            mloss_cst    = (mloss_cst * i + loss_items_cst.detach()) / (i + 1)
            mloss_replay = (mloss_replay * i + loss_replay.detach().unsqueeze(0)) / (i + 1)

            if i % 50 == 0 and is_main:
                mem = f'{torch.cuda.memory_reserved() / 1e9:.2f}G' \
                      if torch.cuda.is_available() else 'cpu'
                log.info(f'  [{epoch:3d}/{EPOCHS-1}][{i:4d}/{nb}]  '
                         f'mem={mem}  '
                         f'cst={mloss_cst.sum().item():.4f}  '
                         f'replay={mloss_replay[0].item():.4f}')

        scheduler.step()

        if is_ddp:
            dist.all_reduce(mloss_cst,    op=dist.ReduceOp.SUM); mloss_cst    /= world_size
            dist.all_reduce(mloss_replay, op=dist.ReduceOp.SUM); mloss_replay /= world_size

        # ── End-of-epoch evaluation ──────────────────────────────────────────
        eval_model = ema.ema if ema is not None else de_parallel(model)

        map50_t3, _, p_t3, r_t3, f1_t3 = evaluate_ddp(
            eval_model, val_loader_t3, device, IMG_SIZE, rank, world_size)
        map50_t1, _, p_t1, r_t1, f1_t1 = evaluate_ddp(
            eval_model, val_loader_t1, device, IMG_SIZE, rank, world_size)

        # Per-stratum T1 evaluation (rank 0 only; all ranks gather)
        stratum_res = evaluate_by_stratum_ddp(
            eval_model, val_loader_t1, device, IMG_SIZE, rank, world_size)

        fm_abs    = map50_t1 - T1_BASELINE
        fm_stage3 = map50_t1 - map50_t2_baseline
        lr_now    = optimizer.param_groups[0]['lr']
        loss_total_val = (mloss_cst.sum() + mloss_replay[0]).item()

        if is_main:
            best_t1 = max(best_t1, map50_t1)

            s_tiny   = stratum_res.get('tiny',   float('nan'))
            s_small  = stratum_res.get('small',  float('nan'))
            s_normal = stratum_res.get('normal', float('nan'))
            s_large  = stratum_res.get('large',  float('nan'))

            log.info(
                f'\nEpoch {epoch:3d}/{EPOCHS-1}  loss={loss_total_val:.4f}\n'
                f'  T3: mAP@0.5={map50_t3:.4f}  P={p_t3:.4f}  R={r_t3:.4f}  F1={f1_t3:.4f}\n'
                f'  T1: mAP@0.5={map50_t1:.4f}  '
                f'fm_abs={fm_abs:.4f}  fm_stage3={fm_stage3:.4f}\n'
                f'  T1 by stratum: tiny={s_tiny:.4f}  small={s_small:.4f}  '
                f'normal={s_normal:.4f}  large={s_large:.4f}\n'
            )

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch,
                    mloss_cst.sum().item(), mloss_replay[0].item(), loss_total_val,
                    map50_t3, p_t3, r_t3, f1_t3,
                    map50_t1, p_t1, r_t1, f1_t1,
                    fm_abs, fm_stage3,
                    s_tiny, s_small, s_normal, s_large,
                    lr_now,
                ])

            ckpt = {
                'epoch':            epoch,
                'best_t3':          best_t3,
                'best_t1':          best_t1,
                'condition':        condition,
                'map50_t2_baseline': map50_t2_baseline,
                'model':            deepcopy(de_parallel(model)).half(),
                'ema':              deepcopy(ema.ema).half(),
                'updates':          ema.updates,
                'optimizer':        optimizer.state_dict(),
                'hyp':              HYP,
                'date':             datetime.now().isoformat(),
            }
            torch.save(ckpt, last_path)

            if save_epoch_ckpts and (epoch % save_epoch_interval == 0 or epoch == EPOCHS - 1):
                torch.save(ckpt, save_dir / 'weights' / f'epoch_{epoch:03d}.pt')

            if map50_t3 > best_t3:
                best_t3          = map50_t3
                fm_at_best_t3    = fm_abs      # record FM at THIS epoch
                fm_s3_at_best_t3 = fm_stage3
                torch.save(ckpt, best_path)
                log.info(f'  ✓ New best T3 mAP@0.5 = {best_t3:.4f}  '
                         f'fm_abs={fm_at_best_t3:.4f}  '
                         f'fm_stage3={fm_s3_at_best_t3:.4f}  '
                         f'(saved {best_path})')

        # ── Early stopping ────────────────────────────────────────────────────
        stop = torch.zeros(1, device=device)
        if is_main and stopper(epoch=epoch, fitness=map50_t3):
            log.info('Early stopping triggered.')
            stop.fill_(1)
        if is_ddp: dist.broadcast(stop, src=0)
        if stop.item(): break

    # ── Final report ──────────────────────────────────────────────────────────
    if is_main:
        log.info(f'\n{"=" * 65}')
        log.info(f'Stage 3 ({condition}) complete.')
        log.info(f'Best T3 mAP@0.5 (CST)       = {best_t3:.4f}')
        log.info(f'mAP_T1_after_T1 (ceiling)   = {T1_BASELINE:.4f}')
        log.info(f'mAP_T1_after_T2 (Stage 3 ref)= {map50_t2_baseline:.4f}')
        log.info(f'fm_abs   at best T3 epoch   = {fm_at_best_t3:.4f}  '
                 f'(vs T1 ceiling)')
        log.info(f'fm_stage3 at best T3 epoch  = {fm_s3_at_best_t3:.4f}  '
                 f'(vs Stage 2 retention)')
        log.info(f'{"=" * 65}')

        (save_dir / 'stage3_mAP.txt').write_text(f'{best_t3:.6f}\n')
        (save_dir / 'stage3_fm.txt').write_text(
            f'fm_abs={fm_at_best_t3:.6f}\n'
            f'fm_stage3={fm_s3_at_best_t3:.6f}\n'
            f'map50_t2_baseline={map50_t2_baseline:.6f}\n'
            f't1_baseline={T1_BASELINE:.6f}\n'
        )
        (save_dir / 'stage3_condition.txt').write_text(f'{condition}\n')
        log.info(f'Results: {csv_path}')

    if is_ddp:
        dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Stage 3: CST fine-tuning — naive / random_stratified / herding')
    p.add_argument('--weights',             type=str,
                   default=str(DEFAULT_STAGE2_WEIGHTS))
    p.add_argument('--replay-buffer',       type=str, default=None,
                   help='Path to replay buffer .pt  '
                        '(required for random_stratified and herding modes).')
    p.add_argument('--replay-mode',         type=str, default='none',
                   choices=['none', 'random_stratified', 'herding'],
                   help='none = naive baseline; '
                        'random_stratified = equal strata, random exemplar selection; '
                        'herding = equal strata, greedy feature-space selection.')
    p.add_argument('--replay-weight',       type=float, default=REPLAY_WEIGHT,
                   help='Loss weight applied to replay loss to compensate for '
                        'smaller replay batch size. Default: 4.0')
    p.add_argument('--epochs',              type=int, default=EPOCHS)
    p.add_argument('--batch-size',          type=int, default=BATCH_SIZE)
    p.add_argument('--workers',             type=int, default=WORKERS)
    p.add_argument('--device',              type=str, default='')
    p.add_argument('--seed',                type=int, default=SEED)
    p.add_argument('--name',                type=str, default='stage3',
                   help='Run name. Use "naive", "random_stratified", "herding".')
    p.add_argument('--save-epoch-ckpts',    action='store_true')
    p.add_argument('--save-epoch-interval', type=int, default=5)
    p.add_argument('--cst-root',            type=str, default=str(CST_ROOT))
    p.add_argument('--rgbt-root',           type=str, default=str(RGBT_ROOT))
    p.add_argument('--t1-baseline',         type=float, default=T1_BASELINE)
    # Kept for backward compat; --replay-mode none is preferred
    p.add_argument('--no-replay',           action='store_true',
                   help='Deprecated: use --replay-mode none instead.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    EPOCHS        = args.epochs
    BATCH_SIZE    = args.batch_size
    WORKERS       = args.workers
    SEED          = args.seed
    T1_BASELINE   = args.t1_baseline
    CST_ROOT      = Path(args.cst_root)
    RGBT_ROOT     = Path(args.rgbt_root)

    replay_mode = 'none' if args.no_replay else args.replay_mode
    replay_path = Path(args.replay_buffer) if args.replay_buffer else None

    if replay_mode != 'none' and replay_path is None:
        raise ValueError(
            f'--replay-buffer is required when --replay-mode={replay_mode}')

    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
    RANK       = int(os.environ.get('RANK',       -1))
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE',  1))

    if LOCAL_RANK != -1:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    else:
        device = select_device(args.device)
        RANK = -1

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    if RANK in (-1, 0):
        save_dir = Path(increment_path(SAVE_ROOT / args.name, exist_ok=False))
    else:
        save_dir = None

    if WORLD_SIZE > 1:
        obj = [str(save_dir)]
        dist.broadcast_object_list(obj, src=0)
        save_dir = Path(obj[0])

    train(save_dir, device,
          stage2_weights=Path(args.weights),
          replay_buffer_path=replay_path,
          replay_mode=replay_mode,
          replay_weight=args.replay_weight,
          save_epoch_ckpts=args.save_epoch_ckpts,
          save_epoch_interval=args.save_epoch_interval,
          rank=RANK, local_rank=LOCAL_RANK, world_size=WORLD_SIZE)
