"""
antiuav410.py
-------------
PyTorch Dataset for Anti-UAV410.

Dataset layout
--------------
{root}/
  {split}/
    {sequence}/
      {sequence}_0001.jpg   per-frame JPEG images
      {sequence}_0002.jpg
      ...
      IR_label.json         {"exist": [...], "gt_rect": [[x,y,w,h], ...]}

Usage
-----
from datasets.antiuav410 import AntiUAV410Dataset

ds = AntiUAV410Dataset(
    root  = '/projects/prjs2041/datasets/Anti-UAV410',
    split = 'train',
)
"""

import json
import cv2
import numpy as np
from pathlib import Path

from .base import BaseUAVDataset


class AntiUAV410Dataset(BaseUAVDataset):

    IMG_W = 640
    IMG_H = 512

    def __init__(self, root, split='train', transform=None, skip_empty=False):
        """
        Args:
            root        : path to Anti-UAV410 root directory
            split       : 'train', 'val', or 'test'
            transform   : optional transform applied to uint8 BGR numpy frame
            skip_empty  : if True, frames with exist=0 are excluded from index
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
            ann_path = seq_dir / 'IR_label.json'
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

                # Anti-UAV410 frame filenames: {seq_name}_{frame:04d}.jpg
                # Frame index is 1-based in filenames
                img_name = f'{seq_dir.name}_{frame_idx + 1:04d}.jpg'
                img_path = seq_dir / img_name

                self._index.append({
                    'seq':      seq_dir.name,
                    'img_path': str(img_path),
                    'frame':    frame_idx,
                    'exist':    int(e),
                    'box':      box,
                    'split':    self.split,
                })

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _load_frame(self, entry):
        img = cv2.imread(entry['img_path'])
        if img is None:
            # Return blank frame if file missing (graceful degradation)
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
