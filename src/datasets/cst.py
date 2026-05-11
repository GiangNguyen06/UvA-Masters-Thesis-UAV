"""
cst.py
------
PyTorch Dataset for CST Anti-UAV.

Dataset layout
--------------
{root}/
  train/
    {sequence}/
      {frame}.jpg           per-frame JPEG images (1-based naming)
      IR_label.json         {"exist": [...], "gt_rect": [[x,y,w,h], ...]}
  val/
    {sequence}/
      ...
  test/
    {sequence}/
      {frame}.jpg
      gt.txt                one line per frame: "x y w h" or "0 0 0 0" if absent
                            (some test sequences use gt.txt instead of IR_label.json)

Usage
-----
from datasets.cst import CSTDataset

ds = CSTDataset(
    root  = '/projects/prjs2041/datasets/CST-AntiUAV',
    split = 'train',
)
"""

import json
import cv2
import numpy as np
from pathlib import Path

from .base import BaseUAVDataset


class CSTDataset(BaseUAVDataset):

    IMG_W = 640
    IMG_H = 512

    def __init__(self, root, split='train', transform=None, skip_empty=False):
        """
        Args:
            root        : path to CST Anti-UAV root directory
            split       : 'train', 'val', or 'test'
            transform   : optional transform applied to uint8 BGR numpy frame
            skip_empty  : if True, frames with exist=0 / all-zero gt are excluded
        """
        self.root       = Path(root)
        self.split      = split
        self.skip_empty = skip_empty
        super().__init__(transform=transform)

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self):
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f'Split directory not found: {split_dir}')

        sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for seq_dir in sequences:
            # Prefer IR_label.json; fall back to gt.txt
            if (seq_dir / 'IR_label.json').exists():
                self._index_from_json(seq_dir)
            elif (seq_dir / 'gt.txt').exists():
                self._index_from_gt_txt(seq_dir)
            else:
                print(f'[WARN] No annotation file found in {seq_dir}, skipping.')

    def _index_from_json(self, seq_dir):
        with open(seq_dir / 'IR_label.json') as f:
            data = json.load(f)

        exist   = data.get('exist', [])
        gt_rect = data.get('gt',   data.get('gt_rect', []))  # CST uses 'gt', others use 'gt_rect'

        if len(exist) != len(gt_rect):
            print(f'[WARN] Length mismatch in {seq_dir.name}, skipping.')
            return

        for frame_idx, (e, box) in enumerate(zip(exist, gt_rect)):
            if self.skip_empty and e == 0:
                continue

            img_path = self._find_image(seq_dir, frame_idx)
            self._index.append({
                'seq':      seq_dir.name,
                'img_path': str(img_path) if img_path else None,
                'frame':    frame_idx,
                'exist':    int(e),
                'box':      box,
                'split':    self.split,
                'ann_fmt':  'json',
            })

    def _index_from_gt_txt(self, seq_dir):
        with open(seq_dir / 'gt.txt') as f:
            lines = [l.strip() for l in f if l.strip()]

        for frame_idx, line in enumerate(lines):
            parts = line.replace(',', ' ').split()
            if len(parts) < 4:
                continue

            x, y, w, h = float(parts[0]), float(parts[1]), \
                         float(parts[2]), float(parts[3])
            e = 0 if (w == 0 and h == 0) else 1

            if self.skip_empty and e == 0:
                continue

            img_path = self._find_image(seq_dir, frame_idx)
            self._index.append({
                'seq':      seq_dir.name,
                'img_path': str(img_path) if img_path else None,
                'frame':    frame_idx,
                'exist':    e,
                'box':      [x, y, w, h] if e == 1 else None,
                'split':    self.split,
                'ann_fmt':  'gt_txt',
            })

    @staticmethod
    def _find_image(seq_dir, frame_idx):
        """
        Try common CST naming conventions (1-based, zero-padded to 4 or 5 digits).
        Returns the first path that exists, or None.
        """
        candidates = [
            seq_dir / f'{frame_idx + 1:06d}.jpg',  # CST uses 6-digit: 000001.jpg
            seq_dir / f'{frame_idx + 1:05d}.jpg',
            seq_dir / f'{frame_idx + 1:04d}.jpg',
            seq_dir / f'{frame_idx:06d}.jpg',
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _load_frame(self, entry):
        img_path = entry.get('img_path')
        if img_path and Path(img_path).exists():
            img = cv2.imread(img_path)
        else:
            img = None

        if img is None:
            img = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)

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

        return img, labels, None
