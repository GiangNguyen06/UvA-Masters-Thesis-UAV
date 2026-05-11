"""
antiuav_rgbt.py
---------------
PyTorch Dataset for Anti-UAV-RGBT.

Dataset layout
--------------
{root}/
  {split}/
    {sequence}/
      infrared.mp4      IR video  (640×512)
      visible.mp4       RGB video (640×512, same length)
      infrared.json     {"exist": [...], "gt_rect": [[x,y,w,h], ...]}

Usage
-----
from datasets.antiuav_rgbt import AntiUAVRGBTDataset

ds = AntiUAVRGBTDataset(
    root   = '/projects/prjs2041/datasets/Anti-UAV-RGBT',
    split  = 'train',       # 'train' | 'val' | 'test'
    return_rgb = False,     # set True to also return visible frame
)
sample = ds[0]
# sample['image']   (3, 512, 640) float32 IR frame
# sample['labels']  (N, 5)        [cls xc yc w h] normalised
# sample['rgb']     (3, 512, 640) if return_rgb=True
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path

from .base import BaseUAVDataset


class AntiUAVRGBTDataset(BaseUAVDataset):

    IMG_W = 640
    IMG_H = 512

    def __init__(self, root, split='train', transform=None, return_rgb=False,
                 skip_empty=False):
        """
        Args:
            root        : path to Anti-UAV-RGBT root directory
            split       : 'train', 'val', or 'test'
            transform   : optional transform applied to uint8 BGR numpy frame
            return_rgb  : also load and return the paired visible (RGB) frame
            skip_empty  : if True, frames with exist=0 are excluded from index
        """
        self.root       = Path(root)
        self.split      = split
        self.skip_empty = skip_empty
        # VideoCapture handles are opened lazily per-sequence and cached
        self._ir_caps   = {}   # seq_name -> cv2.VideoCapture (IR)
        self._rgb_caps  = {}   # seq_name -> cv2.VideoCapture (visible)
        super().__init__(transform=transform, return_rgb=return_rgb)

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self):
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f'Split directory not found: {split_dir}')

        sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for seq_dir in sequences:
            ann_path = seq_dir / 'infrared.json'
            if not ann_path.exists():
                continue

            with open(ann_path) as f:
                data = json.load(f)

            exist   = data.get('exist',   [])
            gt_rect = data.get('gt_rect', [])

            if len(exist) != len(gt_rect):
                print(f'[WARN] Length mismatch in {seq_dir.name}, skipping.')
                continue

            for frame_idx, (e, box) in enumerate(zip(exist, gt_rect)):
                if self.skip_empty and e == 0:
                    continue
                self._index.append({
                    'seq':       seq_dir.name,
                    'seq_dir':   str(seq_dir),
                    'frame':     frame_idx,
                    'exist':     int(e),
                    'box':       box,   # [x, y, w, h] pixels or None
                    'split':     self.split,
                })

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _get_cap(self, cache, seq_dir, filename):
        """Return a cached VideoCapture, opening it on first access."""
        if seq_dir not in cache:
            video_path = Path(seq_dir) / filename
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f'Cannot open video: {video_path}')
            cache[seq_dir] = cap
        return cache[seq_dir]

    def _read_frame(self, cap, frame_idx):
        """Seek to frame_idx and return the decoded BGR frame."""
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f'Failed to read frame {frame_idx}')
        return frame

    def _load_frame(self, entry):
        cap = self._get_cap(self._ir_caps, entry['seq_dir'], 'infrared.mp4')
        img = self._read_frame(cap, entry['frame'])

        # Build label
        e, box = entry['exist'], entry['box']
        if e == 1 and box is not None and len(box) == 4:
            x, y, w, h = box
            if w > 0 and h > 0:
                xc, yc, wn, hn = self.xywh_to_yolo(x, y, w, h,
                                                    self.IMG_W, self.IMG_H)
                labels = np.array([[0, xc, yc, wn, hn]], dtype=np.float32)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        return img, labels, None   # no motion mask for this dataset

    # ── Override __getitem__ to optionally add RGB ────────────────────────────

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        if self.return_rgb:
            entry = self._index[idx]
            cap = self._get_cap(self._rgb_caps, entry['seq_dir'], 'visible.mp4')
            rgb_bgr = self._read_frame(cap, entry['frame'])
            out['rgb'] = torch.from_numpy(
                rgb_bgr[:, :, ::-1].copy()
            ).permute(2, 0, 1).float() / 255.0

        return out

    def __del__(self):
        for cap in self._ir_caps.values():
            cap.release()
        for cap in self._rgb_caps.values():
            cap.release()
