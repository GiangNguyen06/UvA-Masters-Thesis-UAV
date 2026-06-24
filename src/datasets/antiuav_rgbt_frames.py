"""
antiuav_rgbt_frames.py
-----------------------
Fast drop-in replacement for AntiUAVRGBTDataset that reads pre-extracted
JPEG frames instead of seeking inside mp4 files.

Why this matters
----------------
cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, idx) forces the decoder to seek
from the nearest keyframe and then decode forward — on GPFS/NFS this costs
~0.4–0.6 s per random-access batch even with 4 workers.  Pre-extracted JPEGs
reduce each __getitem__ call to a single cv2.imread (~1–2 ms on local NVMe,
~5–10 ms on GPFS), bringing batch I/O time under 0.05 s.

Frame layout (produced by extract_frames_antiuav_rgbt.py)
----------------------------------------------------------
{frames_root}/{split}/{seq}/ir_{frame:06d}.jpg

Usage
-----
from datasets.antiuav_rgbt_frames import AntiUAVRGBTFramesDataset

ds = AntiUAVRGBTFramesDataset(
    frames_root = '/scratch-local/$USER.$JOB_ID/Anti-UAV-RGBT-frames',
    ann_root    = '/scratch-local/$USER.$JOB_ID/Anti-UAV-RGBT',  # for JSON
    split       = 'train',
)

Fallback
--------
If frames_root is None or does not contain the expected layout, the class
falls back transparently to VideoCapture-based loading (slower but correct).
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path

from .base import BaseUAVDataset


class AntiUAVRGBTFramesDataset(BaseUAVDataset):
    """
    Fast dataset for Anti-UAV-RGBT using pre-extracted JPEG frames.

    Parameters
    ----------
    frames_root : str | Path
        Root of pre-extracted frames (output of extract_frames_antiuav_rgbt.py).
        If None, falls back to VideoCapture (uses ann_root for both JSON and video).
    ann_root : str | Path
        Root of the original Anti-UAV-RGBT dataset (needed for JSON annotations
        and as VideoCapture fallback).
    split : str
        'train' | 'val' | 'test'
    transform : callable, optional
        Applied to raw uint8 BGR numpy frame.
    return_rgb : bool
        If True, also load RGB visible frame (always via VideoCapture — only IR
        is pre-extracted).
    skip_empty : bool
        Exclude frames where exist == 0.
    """

    IMG_W = 640
    IMG_H = 512

    def __init__(self, frames_root, ann_root, split='train',
                 transform=None, return_rgb=False, skip_empty=False):
        self.frames_root = Path(frames_root) if frames_root is not None else None
        self.ann_root    = Path(ann_root)
        self.split       = split
        self.skip_empty  = skip_empty

        # VideoCapture fallback caches (used only when JPEG not found)
        self._ir_caps  = {}
        self._rgb_caps = {}

        # Check if frames are available
        self._use_frames = self._check_frames_available()
        if self._use_frames:
            print(f'[AntiUAVRGBTFramesDataset] Using pre-extracted JPEGs from {self.frames_root}')
        else:
            print(f'[AntiUAVRGBTFramesDataset] JPEG frames not found — falling back to VideoCapture')

        super().__init__(transform=transform, return_rgb=return_rgb)

    # ── Frame availability check ──────────────────────────────────────────────

    def _check_frames_available(self):
        """Return True if the frames_root looks populated."""
        if self.frames_root is None:
            return False
        split_dir = self.frames_root / self.split
        if not split_dir.exists():
            return False
        # Check at least one sequence directory with at least one JPEG exists
        for seq_dir in split_dir.iterdir():
            if seq_dir.is_dir():
                jpegs = list(seq_dir.glob('ir_*.jpg'))
                if jpegs:
                    return True
        return False

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self):
        split_dir = self.ann_root / self.split
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
                    'seq_dir':   str(seq_dir),     # original dir (video + JSON)
                    'frame':     frame_idx,
                    'exist':     int(e),
                    'box':       box,
                    'split':     self.split,
                })

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _jpeg_path(self, entry):
        """Return Path to pre-extracted JPEG for this entry."""
        return (self.frames_root / entry['split'] /
                entry['seq'] / f"ir_{entry['frame']:06d}.jpg")

    def _load_frame_jpeg(self, entry):
        """Fast path: read JPEG directly."""
        p = self._jpeg_path(entry)
        frame = cv2.imread(str(p))
        if frame is None:
            raise IOError(f'Failed to read JPEG: {p}')
        return frame

    def _get_cap(self, cache, seq_dir, filename):
        if seq_dir not in cache:
            video_path = Path(seq_dir) / filename
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f'Cannot open video: {video_path}')
            cache[seq_dir] = cap
        return cache[seq_dir]

    def _read_frame_cap(self, cap, frame_idx):
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f'Failed to read frame {frame_idx}')
        return frame

    def _load_frame(self, entry):
        # Choose fast JPEG path or VideoCapture fallback
        if self._use_frames:
            try:
                frame = self._load_frame_jpeg(entry)
            except IOError:
                # Individual frame missing/corrupt — try VideoCapture if mp4
                # exists (original dataset layout), otherwise return a black
                # frame.  A handful of black frames out of 211K is negligible
                # for training; crashing is not acceptable.
                video_path = Path(entry['seq_dir']) / 'infrared.mp4'
                if video_path.exists():
                    cap   = self._get_cap(self._ir_caps, entry['seq_dir'], 'infrared.mp4')
                    frame = self._read_frame_cap(cap, entry['frame'])
                else:
                    print(f'[WARN] Missing JPEG and no mp4 fallback — '
                          f'black frame: {self._jpeg_path(entry)}')
                    frame = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        else:
            cap   = self._get_cap(self._ir_caps, entry['seq_dir'], 'infrared.mp4')
            frame = self._read_frame_cap(cap, entry['frame'])

        # Build YOLO-format label
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

        return frame, labels, None   # no motion mask

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        if self.return_rgb:
            entry = self._index[idx]
            cap   = self._get_cap(self._rgb_caps, entry['seq_dir'], 'visible.mp4')
            rgb_bgr = self._read_frame_cap(cap, entry['frame'])
            out['rgb'] = torch.from_numpy(
                rgb_bgr[:, :, ::-1].copy()
            ).permute(2, 0, 1).float() / 255.0

        return out

    def __del__(self):
        for cap in self._ir_caps.values():
            cap.release()
        for cap in self._rgb_caps.values():
            cap.release()
