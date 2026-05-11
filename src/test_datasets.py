#!/usr/bin/env python3
"""
test_datasets.py
----------------
Smoke-test for all four UAV dataset classes.

Checks that:
  1. Dataset builds its index without crashing
  2. First item loads and has the right tensor shapes
  3. DataLoader yields a batch with collate_fn

Run on Snellius:
  /home/knguyen1/.conda/envs/uav_master/bin/python test_datasets.py
"""

import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/projects/prjs2041/uav_code')
from datasets import (
    AntiUAVRGBTDataset,
    AntiUAV410Dataset,
    ARD100Dataset,
    CSTDataset,
)
from datasets.base import BaseUAVDataset

# ── Dataset roots ─────────────────────────────────────────────────────────────
ROOTS = {
    'AntiUAV-RGBT': '/projects/prjs2041/datasets/Anti-UAV-RGBT',
    'AntiUAV410':   '/projects/prjs2041/datasets/Anti-UAV410',
    'ARD100':       '/projects/prjs2041/datasets/ARD100',
    'CST':          '/projects/prjs2041/datasets/CST-AntiUAV',
}

MASK_ROOT = '/projects/prjs2041/datasets/ARD100/masks_npz'  # may not exist yet


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_sample(name, sample):
    img = sample['image']
    lbl = sample['labels']
    msk = sample['mask']
    print(f'  image  : {tuple(img.shape)}  dtype={img.dtype}  '
          f'min={img.min():.3f}  max={img.max():.3f}')
    print(f'  labels : {tuple(lbl.shape)}')
    if msk is not None:
        print(f'  mask   : {tuple(msk.shape)}')
    else:
        print(f'  mask   : None')
    print(f'  meta   : {sample["meta"]}')
    assert img.shape[0] == 3,          f'{name}: expected 3 channels'
    assert img.dtype == torch.float32, f'{name}: expected float32'
    assert 0.0 <= img.min() and img.max() <= 1.0, f'{name}: image out of [0,1]'
    assert lbl.ndim == 2 and lbl.shape[1] == 5,   f'{name}: labels wrong shape'


def check_loader(name, ds):
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=BaseUAVDataset.collate_fn, num_workers=0)
    batch = next(iter(loader))
    imgs   = batch['images']
    labels = batch['labels']
    print(f'  batch images : {tuple(imgs.shape)}')
    print(f'  batch labels : {tuple(labels.shape)}  '
          f'(cols: batch_idx cls xc yc w h)')


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_antiuav_rgbt():
    print('\n── Anti-UAV-RGBT (test split) ──')
    ds = AntiUAVRGBTDataset(root=ROOTS['AntiUAV-RGBT'], split='test')
    print(f'  Index size: {len(ds)}')
    sample = ds[0]
    check_sample('AntiUAV-RGBT', sample)
    check_loader('AntiUAV-RGBT', ds)
    print('  PASSED')


def test_antiuav410():
    print('\n── Anti-UAV410 (train split) ──')
    ds = AntiUAV410Dataset(root=ROOTS['AntiUAV410'], split='train')
    print(f'  Index size: {len(ds)}')
    sample = ds[0]
    check_sample('AntiUAV410', sample)
    check_loader('AntiUAV410', ds)
    print('  PASSED')


def test_ard100():
    print('\n── ARD100 (train split) ──')
    from pathlib import Path
    mask_root = MASK_ROOT if Path(MASK_ROOT).exists() else None
    if mask_root is None:
        print('  [INFO] No mask_root found — testing without motion masks')
    else:
        print(f'  [INFO] mask_root = {mask_root}')
    ds = ARD100Dataset(root=ROOTS['ARD100'], split='train', mask_root=mask_root)
    print(f'  Index size: {len(ds)}')
    sample = ds[0]
    check_sample('ARD100', sample)
    # Verify mask is loaded and non-zero when mask_root is provided
    if mask_root is not None:
        msk = sample['mask']
        assert msk is not None, 'ARD100: mask is None despite mask_root being set'
        assert msk.ndim == 3 and msk.shape[0] == 1, \
            f'ARD100: expected (1, H, W) mask, got {tuple(msk.shape)}'
        # Frame 0 is all-zeros by design (FD5 diffs frame t with t-5;
        # no previous frames exist at t=0). Sample frame 10 instead.
        sample10 = ds[10]
        msk10 = sample10['mask']
        assert msk10 is not None and msk10.max() > 0, \
            'ARD100: mask at frame 10 is all zeros — FD5 values not loaded correctly'
        print(f'  mask non-zero check (frame 10): PASSED  '
              f'(max={msk10.max():.4f}  mean={msk10.float().mean():.4f})')
    check_loader('ARD100', ds)
    print('  PASSED')


def test_cst():
    print('\n── CST Anti-UAV (train split) ──')
    ds = CSTDataset(root=ROOTS['CST'], split='train')
    print(f'  Index size: {len(ds)}')
    sample = ds[0]
    check_sample('CST', sample)
    check_loader('CST', ds)
    print('  PASSED')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        ('Anti-UAV-RGBT', test_antiuav_rgbt),
        ('Anti-UAV410',   test_antiuav410),
        ('ARD100',        test_ard100),
        ('CST',           test_cst),
    ]

    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            print(f'  FAILED: {e}')
            failed.append((name, str(e)))

    print('\n' + '=' * 50)
    print(f'Results: {len(passed)}/{len(tests)} passed')
    if failed:
        print('Failed:')
        for name, err in failed:
            print(f'  {name}: {err}')
    print('=' * 50)
