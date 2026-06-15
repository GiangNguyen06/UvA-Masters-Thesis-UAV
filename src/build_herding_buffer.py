#!/usr/bin/env python3
"""
build_herding_buffer.py
-----------------------
Build a Scale-Stratified exemplar replay buffer from the Anti-UAV-RGBT
validation set, using the STAGE 2 model's feature space.

Two selection modes (--mode):

  herding  (default)
    Uses greedy iCaRL-style herding: iteratively select the sample whose
    embedding is closest to the current stratum mean, until K exemplars
    are chosen per stratum.  Produces herding_buffer.pt.

  random
    Selects K exemplars per stratum uniformly at random (no feature-space
    logic).  Produces herding_buffer_random.pt.
    Used as the ablation condition in train_stage3.py (--replay-mode
    random_stratified) to isolate the contribution of herding vs.
    simply having balanced strata.

IMPORTANT — use Stage 2 weights, not Stage 1:
  The exemplars selected here will be replayed to a model that starts from
  Stage 2.  Using Stage 2 features for selection means the chosen exemplars
  are representative of the distribution as the *current* model sees it,
  which is the theoretical requirement for herding (iCaRL, 2017).

Algorithm
---------
1. Load the Stage 2 EMA model (best.pt from runs/stage2/seed42).
2. For each frame in the Anti-UAV-RGBT VAL set where a UAV is present,
   run a forward pass and extract the neck P3 embedding (global avg-pool,
   shape [128]).
3. Assign each frame to a UAV size stratum based on GT box area:
     tiny   : side < 16 px  (area < 256 px²)
     small  : side 16-32 px (area 256-1024 px²)
     normal : side 32-64 px (area 1024-4096 px²)
     large  : side > 64 px  (area > 4096 px²)
4. Within each stratum, select K exemplars (herding or random).
5. Save the buffer as a .pt file.

Usage (Snellius):
  # Herding buffer (thesis contribution)
  python build_herding_buffer.py \\
      --weights   /projects/prjs2041/runs/stage2/seed42/weights/best.pt \\
      --data-root $TMPDIR/Anti-UAV-RGBT \\
      --k-per-stratum 75 \\
      --mode herding \\
      --out /projects/prjs2041/runs/stage2/seed42/herding_buffer.pt

  # Random stratified buffer (ablation)
  python build_herding_buffer.py \\
      --weights   /projects/prjs2041/runs/stage2/seed42/weights/best.pt \\
      --data-root $TMPDIR/Anti-UAV-RGBT \\
      --k-per-stratum 75 \\
      --mode random \\
      --out /projects/prjs2041/runs/stage2/seed42/herding_buffer_random.pt

Outputs:
  <out>.pt    — exemplar buffer (loaded by train_stage3.py)
  <out>.txt   — per-stratum counts and selection summary
"""

import sys
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT = Path('/projects/prjs2041/YOLOMG')
UAV_CODE    = Path('/projects/prjs2041/uav_code')
sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

from utils.torch_utils import select_device
from datasets import AntiUAVRGBTDataset

# ── Size bins (UAV-appropriate, matching eval_full_analysis.py) ────────────────
SIZE_BINS = [
    ('tiny',   0,      16**2),
    ('small',  16**2,  32**2),
    ('normal', 32**2,  64**2),
    ('large',  64**2,  float('inf')),
]

IMG_W    = 640   # Anti-UAV-RGBT native width
IMG_H    = 512   # Anti-UAV-RGBT native height
IMG_SIZE = 640

DEFAULT_STAGE2_WEIGHTS = (
    '/projects/prjs2041/runs/stage2/seed42/weights/best.pt'
)


def size_category(wn: float, hn: float) -> str:
    """Assign a normalised (w, h) GT box to a size stratum."""
    w_px = wn * IMG_W
    h_px = hn * IMG_H
    area = w_px * h_px
    for name, lo, hi in SIZE_BINS:
        if lo <= area < hi:
            return name
    return 'large'


# ══════════════════════════════════════════════════════════════════════════════
# Dataset wrapper (only frames where UAV exists)
# ══════════════════════════════════════════════════════════════════════════════

def _letterbox_tensor(img: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = img.shape
    scale   = size / max(h, w)
    nh, nw  = int(round(h * scale)), int(round(w * scale))
    img = F.interpolate(img.unsqueeze(0), size=(nh, nw),
                        mode='bilinear', align_corners=False).squeeze(0)
    ph, pw = size - nh, size - nw
    img = F.pad(img, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=0.5)
    return img


class HerdingDataset(Dataset):
    """
    Wraps AntiUAVRGBTDataset, yielding only frames where exist == 1.
    Returns (img_tensor, img2_zeros, labels, meta) for each frame.
    """

    def __init__(self, root: Path, imgsz: int = 640):
        self.ds    = AntiUAVRGBTDataset(root=root, split='val', skip_empty=True)
        self.imgsz = imgsz

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = _letterbox_tensor(sample['image'], self.imgsz)
        img2   = torch.zeros_like(img)
        labels = sample['labels']   # (N, 5) [cls, xc, yc, w, h]
        meta   = sample.get('meta', {})
        return img, img2, labels, meta

    @staticmethod
    def collate_fn(batch):
        imgs, imgs2, labels_list, metas = zip(*batch)
        imgs  = torch.stack(imgs)
        imgs2 = torch.stack(imgs2)
        return imgs, imgs2, list(labels_list), list(metas)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ══════════════════════════════════════════════════════════════════════════════

def register_embedding_hook(model):
    """
    Register a forward hook on the FPN neck P3 output (layer 20 in YOLOMG,
    128-channel feature map). Global-average-pool → [128]-dim embedding.
    Falls back to the first C3 block if layer 20 is not found.
    """
    embeddings = []

    def hook_fn(module, input, output):
        emb = output.mean(dim=[2, 3])   # (B, C, H, W) → (B, C)
        embeddings.append(emb.detach().cpu())

    target_layer = None
    for name, module in model.named_modules():
        if 'model.20' in name and hasattr(module, 'cv1'):
            target_layer = module
            break

    if target_layer is None:
        for name, module in model.named_modules():
            if hasattr(module, 'cv1') and hasattr(module, 'cv2'):
                target_layer = module
                break

    if target_layer is None:
        raise RuntimeError('Could not find a suitable layer to hook for embeddings.')

    handle = target_layer.register_forward_hook(hook_fn)
    return embeddings, handle


# ══════════════════════════════════════════════════════════════════════════════
# Exemplar selection
# ══════════════════════════════════════════════════════════════════════════════

def greedy_herding(embeddings: np.ndarray, k: int) -> list:
    """
    Select k exemplars from embeddings (N, D) using greedy herding:
    iteratively pick the sample whose addition keeps the running mean
    closest to the full stratum mean.
    Returns a list of k selected indices.
    """
    n = len(embeddings)
    if k >= n:
        return list(range(n))

    full_mean    = embeddings.mean(axis=0)
    selected     = []
    current_mean = np.zeros(embeddings.shape[1], dtype=np.float64)

    for _ in range(k):
        remaining = [i for i in range(n) if i not in selected]
        best_idx  = None
        best_dist = float('inf')

        for i in remaining:
            n_sel     = len(selected)
            cand_mean = (current_mean * n_sel + embeddings[i]) / (n_sel + 1)
            d = np.linalg.norm(cand_mean - full_mean)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        selected.append(best_idx)
        n_sel        = len(selected)
        current_mean = (current_mean * (n_sel - 1) + embeddings[best_idx]) / n_sel

    return selected


def random_selection(n_avail: int, k: int, seed: int = 42) -> list:
    """
    Select k indices from [0, n_avail) uniformly at random.
    Used for the random_stratified ablation condition.
    """
    rng = np.random.default_rng(seed)
    if k >= n_avail:
        return list(range(n_avail))
    return rng.choice(n_avail, k, replace=False).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Build Scale-Stratified replay buffer (herding or random)')
    p.add_argument('--weights',       type=str,
        default=DEFAULT_STAGE2_WEIGHTS,
        help='Stage 2 best.pt to use for feature extraction. '
             'MUST be Stage 2 (not Stage 1) so embeddings match the '
             'model being trained in Stage 3.')
    p.add_argument('--data-root',     type=str,
        default='/gpfs/scratch1/shared/knguyen1/Anti-UAV-RGBT')
    p.add_argument('--k-per-stratum', type=int, default=75,
        help='Exemplars to select per size stratum (default: 75 → 300 total)')
    p.add_argument('--mode',          type=str, default='herding',
        choices=['herding', 'random'],
        help='herding = greedy feature-space selection (thesis contribution); '
             'random  = uniform random selection (ablation baseline).')
    p.add_argument('--seed',          type=int, default=42,
        help='Random seed for --mode random (default: 42)')
    p.add_argument('--out',           type=str, default=None,
        help='Output .pt path. Defaults to <weights_dir>/herding_buffer[_random].pt')
    p.add_argument('--batch-size',    type=int, default=32)
    p.add_argument('--workers',       type=int, default=4)
    p.add_argument('--device',        type=str, default='0')
    p.add_argument('--imgsz',         type=int, default=IMG_SIZE)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.device)

    weights   = Path(args.weights)
    data_root = Path(args.data_root)

    # Default output path reflects selection mode
    if args.out:
        out_path = Path(args.out)
    else:
        suffix   = '' if args.mode == 'herding' else '_random'
        out_path = weights.parent.parent / f'herding_buffer{suffix}.pt'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Scale-Stratified Herding Buffer Builder')
    print(f'  Mode       : {args.mode}')
    print(f'  Weights    : {weights}  (Stage 2)')
    print(f'  Data       : {data_root}')
    print(f'  K/stratum  : {args.k_per_stratum}')
    print(f'  Output     : {out_path}\n')

    # ── Load Stage 2 model ────────────────────────────────────────────────────
    print('Loading Stage 2 model …')
    ckpt  = torch.load(weights, map_location=device, weights_only=False)
    model = (ckpt['ema'] if ckpt.get('ema') else ckpt['model']).float().to(device)
    model.nc = 1; model.names = ['UAV']; model.eval()
    print(f'  Epoch {ckpt.get("epoch","?")}  '
          f'best_fitness={ckpt.get("best_fitness","?")}\n')

    # ── Dataset ───────────────────────────────────────────────────────────────
    print('Loading Anti-UAV-RGBT val (UAV-present frames only) …')
    ds     = HerdingDataset(data_root, args.imgsz)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=False,
                        collate_fn=HerdingDataset.collate_fn)
    print(f'  {len(ds):,} UAV-present frames  |  {len(loader)} batches\n')

    # ── Register embedding hook ───────────────────────────────────────────────
    # Embeddings only needed for herding; still run for random to keep
    # the buffer schema consistent (embeddings field present in both).
    emb_list, hook_handle = register_embedding_hook(model)

    # ── Extract embeddings + metadata ─────────────────────────────────────────
    print('Extracting embeddings …')
    all_embeddings = []
    all_labels     = []
    all_metas      = []
    all_strata     = []

    with torch.no_grad():
        for bi, (imgs, imgs2, labels_list, metas) in enumerate(loader):
            imgs  = imgs.to(device)
            imgs2 = imgs2.to(device)
            _     = model(imgs, imgs2)

            batch_emb = emb_list[-1]   # (B, C)

            for i in range(len(imgs)):
                emb  = batch_emb[i].numpy()
                lbl  = labels_list[i]
                meta = metas[i]

                if len(lbl) > 0:
                    _, _, _, wn, hn = lbl[0]
                    stratum = size_category(float(wn), float(hn))
                else:
                    stratum = 'tiny'   # no GT → treat as tiny (rare)

                all_embeddings.append(emb)
                all_labels.append(lbl)
                all_metas.append(meta)
                all_strata.append(stratum)

            if (bi + 1) % 50 == 0:
                print(f'  [{bi+1}/{len(loader)}] {len(all_embeddings):,} frames processed')

    hook_handle.remove()
    print(f'\n  Total frames with embeddings: {len(all_embeddings):,}')

    # ── Stratum summary ───────────────────────────────────────────────────────
    from collections import Counter
    stratum_counts = Counter(all_strata)
    print('\nStratum distribution:')
    for name, _, _ in SIZE_BINS:
        print(f'  {name:<8}: {stratum_counts.get(name, 0):>6,} frames')

    # ── Per-stratum exemplar selection ────────────────────────────────────────
    mode_label = 'greedy herding' if args.mode == 'herding' else 'random selection'
    print(f'\nRunning {mode_label} (k={args.k_per_stratum} per stratum) …')
    all_embeddings_np = np.stack(all_embeddings)   # (N, D)

    selected_indices = []
    stats_lines      = [
        f'Scale-Stratified Buffer  [{args.mode.upper()}]',
        f'Weights   : {weights}  (Stage 2)',
        f'K/stratum : {args.k_per_stratum}',
        f'Mode      : {args.mode}',
        f'',
        f'{"Stratum":<10} {"Available":>10} {"Selected":>10}',
        '-' * 35,
    ]

    for name, _, _ in SIZE_BINS:
        stratum_idx = [i for i, s in enumerate(all_strata) if s == name]
        n_avail     = len(stratum_idx)
        k           = min(args.k_per_stratum, n_avail)

        if n_avail == 0:
            print(f'  {name}: 0 frames available — skipped')
            stats_lines.append(f'{name:<10} {0:>10} {0:>10}  (none available)')
            continue

        if args.mode == 'herding':
            stratum_embs = all_embeddings_np[stratum_idx]
            local_sel    = greedy_herding(stratum_embs, k)
        else:
            local_sel    = random_selection(n_avail, k, seed=args.seed)

        global_sel = [stratum_idx[i] for i in local_sel]
        selected_indices.extend(global_sel)

        print(f'  {name:<8}: {n_avail:>6,} available → {k} selected')
        stats_lines.append(f'{name:<10} {n_avail:>10,} {k:>10}')

    stats_lines.append('-' * 35)
    stats_lines.append(
        f'{"TOTAL":<10} {len(all_embeddings):>10,} {len(selected_indices):>10}')

    # ── Build buffer ──────────────────────────────────────────────────────────
    print(f'\nBuilding buffer from {len(selected_indices)} exemplars …')

    buf_embeddings = torch.from_numpy(all_embeddings_np[selected_indices])
    buf_labels     = [all_labels[i] for i in selected_indices]
    buf_metas      = [all_metas[i]  for i in selected_indices]
    buf_strata     = [all_strata[i] for i in selected_indices]

    buffer = {
        'embeddings':    buf_embeddings,
        'labels':        buf_labels,
        'metas':         buf_metas,
        'strata':        buf_strata,
        'k_per_stratum': args.k_per_stratum,
        'mode':          args.mode,
        'weights':       str(weights),
        'data_root':     str(data_root),
        'size_bins':     SIZE_BINS,
        'n_total':       len(selected_indices),
    }

    torch.save(buffer, out_path)
    print(f'  Buffer saved : {out_path}')
    print(f'  Buffer size  : {out_path.stat().st_size / 1024:.1f} KB')

    # ── Stats file ────────────────────────────────────────────────────────────
    stats_text = '\n'.join(stats_lines)
    print('\n' + stats_text)
    stats_path = out_path.with_suffix('.txt')
    stats_path.write_text(stats_text)
    print(f'\nStats: {stats_path}')
    print('Done.')


if __name__ == '__main__':
    main()
