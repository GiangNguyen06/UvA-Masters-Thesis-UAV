#!/usr/bin/env python3
"""
train_stage2.py
---------------
Stage 2: Teacher-Student UDA — fine-tune YOLOMG on Anti-UAV410 IR.

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

Motion masks
------------
AntiUAV410 has no ego-motion (stationary IR camera with pre-extracted
JPEG frames), so img2 = zeros, matching Stage 1's treatment of
AntiUAVRGBT.

Usage (testing only — use sbatch for real runs):
  python3 train_stage2.py --weights /projects/prjs2041/runs/stage1/antiuav_rgbt/weights/best.pt

SLURM:
  sbatch run_stage2.sh

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
import time
import csv
import math
import yaml
import random
import logging
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset

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
EPOCHS     = 50           # fine-tuning requires fewer epochs than Stage 1
BATCH_SIZE = 16
WORKERS    = 4
SEED       = 42
NC         = 1
NAMES      = ['UAV']
MODEL_CFG  = str(YOLOMG_ROOT / 'models' / 'dual_uav2.yaml')

# Knowledge distillation weight  (λ_kd in the thesis)
KD_WEIGHT  = 1.0

# NMS thresholds for evaluation
CONF_THRES = 0.001
IOU_THRES  = 0.6
MAP_IOU    = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Dataset wrappers
# ══════════════════════════════════════════════════════════════════════════════

class Stage2Dataset(Dataset):
    """
    Wraps AntiUAV410Dataset for YOLOMG dual-input inference.

    img1 : (3, H, W) float [0,1]  — IR frame
    img2 : (3, H, W) float [0,1]  — zeros (no motion masks; stationary camera)
    labels: (N, 5) [cls, xc, yc, w, h] normalised
    """

    def __init__(self, root: Path, split: str, imgsz: int = 640):
        self.ds    = AntiUAV410Dataset(root=root, split=split)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = sample['image']    # (3, H, W) float [0,1]
        labels = sample['labels']   # (N, 5)
        meta   = sample['meta']

        img  = _letterbox_tensor(img, self.imgsz)
        img2 = torch.zeros_like(img)

        path = f"{meta.get('seq', 'seq')}_{meta.get('frame', idx)}"
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

        return imgs, imgs2, targets, list(paths), list(paths), None, None


class RGBTValDataset(Dataset):
    """
    Wraps AntiUAVRGBTDataset for evaluation — used to track the
    Forgetting Measure on T1 throughout Stage 2 training.
    """

    def __init__(self, root: Path, split: str = 'val', imgsz: int = 640):
        self.ds    = AntiUAVRGBTDataset(root=root, split=split)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = sample['image']
        labels = sample['labels']
        meta   = sample['meta']

        img  = _letterbox_tensor(img, self.imgsz)
        img2 = torch.zeros_like(img)

        path = f"{meta.get('seq', 'seq')}_{meta.get('frame', idx)}"
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

        return imgs, imgs2, targets, list(paths), list(paths), None, None


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


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Distillation loss
# ══════════════════════════════════════════════════════════════════════════════

def kd_loss(student_preds: list, teacher_preds: list) -> torch.Tensor:
    """
    Response-level knowledge distillation: MSE between student and teacher
    raw prediction grids at each detection scale.

    Both student_preds and teacher_preds are lists of 3 tensors
    (P3/P4/P5 scales), each of shape (B, anchors, H_i, W_i, 5+nc).

    The teacher grids encode the T1 knowledge; minimising this distance
    prevents the student from drifting away from the T1 solution.
    """
    loss = torch.tensor(0.0, device=student_preds[0].device)
    for s, t in zip(student_preds, teacher_preds):
        loss = loss + F.mse_loss(s.float(), t.float().detach())
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation  (mAP@0.5)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, imgsz=640):
    """Compute mAP@0.5 and mAP@0.5:0.95 on the given DataLoader."""
    model.eval()
    stats = []

    for imgs, imgs2, targets, *_ in loader:
        imgs    = imgs.to(device)
        imgs2   = imgs2.to(device)
        targets = targets.to(device)

        preds = model(imgs, imgs2)
        # YOLOMG eval mode returns (inference_out, train_out) tuple
        if isinstance(preds, tuple):
            preds = preds[0]             # take decoded inference output
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

            predn = det.clone()
            if nl:
                tbox = xywh2xyxy(gt[:, 1:5]) * imgsz
                tbox = tbox.to(device)
                iou  = box_iou(tbox, predn[:, :4])
                correct = torch.zeros(len(det), 1, dtype=torch.bool, device=device)
                if iou.numel():
                    x = torch.where(iou >= MAP_IOU)
                    if x[0].shape[0]:
                        matches = torch.cat((
                            torch.stack(x, 1),
                            iou[x[0], x[1]][:, None]
                        ), 1).cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), 0] = True
            else:
                correct = torch.zeros(len(det), 1, dtype=torch.bool)

            stats.append((
                correct.cpu(),
                det[:, 4].cpu(),
                det[:, 5].cpu(),
                tcls
            ))

    if not stats:
        model.train()
        return 0.0, 0.0

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        _, _, _, _, _, ap, _ = ap_per_class(*stats, plot=False, names={0: 'UAV'})
        map50 = float(ap[:, 0].mean()) if ap.ndim == 2 else float(ap[0])
        mapxx = float(ap.mean())       if ap.ndim == 2 else float(ap.mean())
    else:
        map50, mapxx = 0.0, 0.0

    model.train()
    return map50, mapxx


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> Model:
    """
    Build a fresh YOLOMG model and load EMA weights from a Stage 1 checkpoint.
    Returns the model in train mode with float32 weights.
    """
    model = Model(MODEL_CFG, ch=3, ch2=3, nc=NC).to(device)
    model.nc    = NC
    model.hyp   = HYP
    model.names = NAMES

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Prefer EMA weights (more stable than raw model weights)
    if 'ema' in ckpt and ckpt['ema'] is not None:
        state_dict = ckpt['ema'].float().state_dict()
        src = 'EMA'
    else:
        state_dict = ckpt['model'].float().state_dict()
        src = 'model'

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[WARN] {len(missing)} missing keys loading from {src}: {missing[:5]}')
    if unexpected:
        print(f'[WARN] {len(unexpected)} unexpected keys loading from {src}')

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(save_dir: Path, device: torch.device, stage1_weights: Path):
    init_seeds(SEED)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'weights').mkdir(exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'train.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    log.info(f'Stage 2 Teacher-Student UDA  →  {save_dir}')
    log.info(f'Stage 1 weights: {stage1_weights}')
    log.info(f'Device: {device}  |  IMG_SIZE={IMG_SIZE}  EPOCHS={EPOCHS}  BS={BATCH_SIZE}')
    log.info(f'KD weight λ = {KD_WEIGHT}')

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(HYP, f)

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info('Loading datasets …')
    train_ds   = Stage2Dataset(ANTIUAV410_ROOT,   split='train', imgsz=IMG_SIZE)
    val_ds_t2  = Stage2Dataset(ANTIUAV410_ROOT,   split='val',   imgsz=IMG_SIZE)
    val_ds_t1  = RGBTValDataset(ANTIUAV_RGBT_ROOT, split='val',  imgsz=IMG_SIZE)
    log.info(f'  T2 train : {len(train_ds):,} frames   (AntiUAV410)')
    log.info(f'  T2 val   : {len(val_ds_t2):,} frames   (AntiUAV410)')
    log.info(f'  T1 val   : {len(val_ds_t1):,} frames   (AntiUAVRGBT — FM tracking)')

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True,
        collate_fn=Stage2Dataset.collate_fn,
    )
    val_loader_t2 = DataLoader(
        val_ds_t2, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=WORKERS, pin_memory=True,
        collate_fn=Stage2Dataset.collate_fn,
    )
    val_loader_t1 = DataLoader(
        val_ds_t1, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=WORKERS, pin_memory=True,
        collate_fn=RGBTValDataset.collate_fn,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    log.info(f'Loading student from {stage1_weights} …')
    student = load_model_from_ckpt(stage1_weights, device)

    log.info('Creating frozen teacher (same weights) …')
    teacher = load_model_from_ckpt(stage1_weights, device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    log.info(f'  Parameters: {sum(p.numel() for p in student.parameters()):,}')
    log.info(f'  Trainable:  {sum(p.numel() for p in student.parameters() if p.requires_grad):,}')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    g0, g1, g2 = [], [], []
    for v in student.modules():
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

    lf = lambda x: ((1 - math.cos(x * math.pi / EPOCHS)) / 2) * (HYP['lrf'] - 1) + 1
    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    # ── AMP + EMA ─────────────────────────────────────────────────────────────
    scaler  = amp.GradScaler(enabled=(device.type != 'cpu'))
    ema     = ModelEMA(student)
    stopper = EarlyStopping(patience=15)

    # ── Loss ──────────────────────────────────────────────────────────────────
    compute_loss = ComputeLoss(student)

    # ── Results CSV ───────────────────────────────────────────────────────────
    csv_path = save_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'loss_box', 'loss_obj', 'loss_cls', 'loss_kd',
            'loss_total', 'mAP50_T2', 'mAP50_T1', 'lr'
        ])

    # ── Training loop ─────────────────────────────────────────────────────────
    nb           = len(train_loader)
    nw           = max(round(HYP['warmup_epochs'] * nb), 50)
    best_t2      = 0.0    # best mAP on AntiUAV410 val (T2)
    best_t1      = 0.0    # best mAP on AntiUAVRGBT val (T1, for FM)
    last_path    = save_dir / 'weights' / 'last.pt'
    best_path    = save_dir / 'weights' / 'best.pt'

    log.info(f'\nStarting Stage 2 training for {EPOCHS} epochs …')
    log.info(f'  L = L_det + {KD_WEIGHT} × L_kd\n')

    for epoch in range(EPOCHS):
        student.train()
        mloss = torch.zeros(4, device=device)   # box, obj, cls, kd

        for i, (imgs, imgs2, targets, *_) in enumerate(train_loader):
            ni    = i + nb * epoch
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

            with amp.autocast(enabled=(device.type != 'cpu')):
                # ── Student forward (train mode → returns raw prediction grids)
                student_preds = student(imgs, imgs2)
                det_loss, loss_items = compute_loss(student_preds, targets)

                # ── Teacher forward (eval mode, no_grad)
                #    Returns (inference_out, train_out); we use train_out
                #    for scale-matched distillation.
                with torch.no_grad():
                    teacher_out = teacher(imgs, imgs2)
                    # In eval mode, YOLOMG returns (pred_nms, train_out)
                    teacher_grids = teacher_out[1] if isinstance(teacher_out, tuple) else teacher_out

                # ── KD loss
                # student_preds is a list of 3 raw grids in training mode
                # teacher_grids should be the same structure
                if isinstance(student_preds, (list, tuple)) and isinstance(teacher_grids, (list, tuple)):
                    kd = kd_loss(list(student_preds), list(teacher_grids))
                else:
                    kd = torch.tensor(0.0, device=device)

                total_loss = det_loss + KD_WEIGHT * kd

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(student)

            # Running mean loss: box, obj, cls, kd
            batch_items = torch.cat([loss_items, kd.unsqueeze(0).detach()])
            mloss = (mloss * i + batch_items) / (i + 1)

            if i % 50 == 0:
                mem = f'{torch.cuda.memory_reserved() / 1e9:.2f}G' \
                      if torch.cuda.is_available() else 'cpu'
                log.info(
                    f'  [{epoch:3d}/{EPOCHS-1}][{i:4d}/{nb}]  '
                    f'mem={mem}  '
                    f'box={mloss[0]:.4f}  obj={mloss[1]:.4f}  '
                    f'cls={mloss[2]:.4f}  kd={mloss[3]:.4f}'
                )

        scheduler.step()

        # ── End-of-epoch evaluation ───────────────────────────────────────────
        t_epoch = datetime.now().strftime('%H:%M:%S')
        map50_t2, _     = evaluate(ema.ema, val_loader_t2, device, IMG_SIZE)
        map50_t1, _     = evaluate(ema.ema, val_loader_t1, device, IMG_SIZE)
        lr_now          = optimizer.param_groups[0]['lr']
        total_loss_val  = mloss.sum().item()

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
                mloss[0].item(), mloss[1].item(), mloss[2].item(), mloss[3].item(),
                total_loss_val, map50_t2, map50_t1, lr_now
            ])

        # Checkpoint
        ckpt = {
            'epoch':        epoch,
            'best_t2':      best_t2,
            'best_t1':      best_t1,
            'model':        deepcopy(de_parallel(student)).half(),
            'ema':          deepcopy(ema.ema).half(),
            'updates':      ema.updates,
            'optimizer':    optimizer.state_dict(),
            'hyp':          HYP,
            'kd_weight':    KD_WEIGHT,
            'date':         datetime.now().isoformat(),
        }
        torch.save(ckpt, last_path)
        if map50_t2 > best_t2:
            best_t2 = map50_t2
            torch.save(ckpt, best_path)
            log.info(f'  ✓ New best T2 mAP@0.5 = {best_t2:.4f}  (saved {best_path})')

        if stopper(epoch=epoch, fitness=map50_t2):
            log.info('Early stopping triggered.')
            break

    # ── Final report ──────────────────────────────────────────────────────────
    log.info(f'\n{"=" * 60}')
    log.info(f'Stage 2 complete.')
    log.info(f'Best T2 mAP@0.5 (AntiUAV410) = {best_t2:.4f}')
    log.info(f'Best T1 mAP@0.5 (AntiUAVRGBT) after T2 = {best_t1:.4f}  ← FM numerator')
    log.info(f'Weights: {best_path}')
    log.info(f'Results: {csv_path}')

    # Write T2 mAP for reporting
    t2_txt = save_dir / 'stage2_t2_mAP.txt'
    t2_txt.write_text(f'{best_t2:.6f}\n')
    log.info(f'T2 mAP written to {t2_txt}')

    # Write T1 mAP after T2 — used for Forgetting Measure
    t1_txt = save_dir / 'stage2_t1_mAP.txt'
    t1_txt.write_text(f'{best_t1:.6f}\n')
    log.info(f'T1 mAP after T2 written to {t1_txt}  ← FM = this − stage1_mAP.txt')


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Stage 2: Teacher-Student UDA on AntiUAV410')
    p.add_argument('--weights',    type=str, default=str(DEFAULT_STAGE1_WEIGHTS),
                   help='Path to Stage 1 best.pt checkpoint')
    p.add_argument('--epochs',     type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--imgsz',      type=int, default=IMG_SIZE)
    p.add_argument('--workers',    type=int, default=WORKERS)
    p.add_argument('--device',     type=str, default='')
    p.add_argument('--kd-weight',  type=float, default=KD_WEIGHT,
                   help='KD loss weight λ (default: 1.0)')
    p.add_argument('--name',       type=str, default='antiuav410')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    IMG_SIZE   = args.imgsz
    WORKERS    = args.workers
    KD_WEIGHT  = args.kd_weight

    device   = select_device(args.device)
    save_dir = Path(increment_path(SAVE_ROOT / args.name, exist_ok=False))

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    train(save_dir, device, stage1_weights=Path(args.weights))
