"""
base.py
-------
BaseUAVDataset: shared interface for all four Anti-UAV dataset classes.

Every subclass must implement:
    _build_index()  ->  populates self._index as a list of dicts
    _load_frame(entry) -> returns (img_bgr: np.ndarray, labels: np.ndarray)

__getitem__ returns a dict:
    {
        'image'  : torch.Tensor  (C, H, W), float32, [0, 1]
        'labels' : torch.Tensor  (N, 5),    float32, [cls, xc, yc, w, h] normalised
        'mask'   : torch.Tensor  (1, H, W) or None   -- motion diff map
        'meta'   : dict  { 'seq': str, 'frame': int, 'exist': int }
    }
"""

import abc
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseUAVDataset(Dataset, abc.ABC):

    IMG_W = 640
    IMG_H = 512

    def __init__(self, transform=None, return_rgb=False):
        """
        Args:
            transform:  optional callable applied to the uint8 BGR numpy frame
                        before conversion to tensor (e.g. albumentations pipeline)
            return_rgb: if True, subclasses that have a visible stream will
                        also populate 'rgb' in the output dict
        """
        self.transform = transform
        self.return_rgb = return_rgb
        self._index = []      # built by subclass _build_index()
        self._build_index()

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _build_index(self):
        """Populate self._index with one dict per sample."""

    @abc.abstractmethod
    def _load_frame(self, entry: dict):
        """
        Load raw data for a single sample.

        Returns
        -------
        img   : np.ndarray  uint8 BGR (H, W, 3)
        labels: np.ndarray  float32 (N, 5) YOLO-format [cls xc yc w h] normalised
                            or (0, 5) empty array if exist=0
        mask  : np.ndarray float32 (H, W) or None
        """

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        entry = self._index[idx]
        img, labels, mask = self._load_frame(entry)

        if self.transform is not None:
            img = self.transform(img)

        # BGR uint8 → RGB float32 tensor (C, H, W) in [0, 1]
        img_t = torch.from_numpy(
            img[:, :, ::-1].copy()           # BGR → RGB
        ).permute(2, 0, 1).float() / 255.0

        labels_t = torch.from_numpy(
            labels if labels.shape[0] > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        mask_t = (
            torch.from_numpy(mask).unsqueeze(0).float()
            if mask is not None else None
        )

        out = {
            'image':  img_t,
            'labels': labels_t,
            'mask':   mask_t,
            'meta':   entry,
        }
        return out

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def xywh_to_yolo(x, y, w, h, img_w, img_h):
        """Pixel [x_tl, y_tl, w, h] → normalised YOLO [xc, yc, w, h]."""
        xc = (x + w / 2.0) / img_w
        yc = (y + h / 2.0) / img_h
        wn = w / img_w
        hn = h / img_h
        return (
            float(np.clip(xc, 0, 1)),
            float(np.clip(yc, 0, 1)),
            float(np.clip(wn, 0, 1)),
            float(np.clip(hn, 0, 1)),
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate that handles variable-length label tensors.
        Adds a batch-index column to labels so they can be stacked:
            (batch_idx, cls, xc, yc, w, h)
        """
        images, masks, metas = [], [], []
        all_labels = []

        for i, sample in enumerate(batch):
            images.append(sample['image'])
            masks.append(sample['mask'])
            metas.append(sample['meta'])

            lbl = sample['labels']           # (N, 5)
            if lbl.shape[0] > 0:
                bi = torch.full((lbl.shape[0], 1), i, dtype=torch.float32)
                all_labels.append(torch.cat([bi, lbl], dim=1))

        images = torch.stack(images, 0)
        labels = (
            torch.cat(all_labels, 0)
            if all_labels
            else torch.zeros((0, 6), dtype=torch.float32)
        )
        masks = (
            torch.stack([m for m in masks if m is not None], 0)
            if any(m is not None for m in masks)
            else None
        )
        return {'images': images, 'labels': labels, 'masks': masks, 'metas': metas}
