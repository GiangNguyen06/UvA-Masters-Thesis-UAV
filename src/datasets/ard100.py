"""
ard100.py
---------
PyTorch Dataset for ARD100.

Dataset layout
--------------
{root}/
  train_videos/
    phantom09.mp4
    phantom10.mp4
    ...
  test_videos/
    phantom02.mp4
    ...
  Annotations/
    phantom09/
      phantom09_0001.xml   Pascal VOC XML per frame
      phantom09_0002.xml
      ...

Motion masks (optional, precomputed)
-------------------------------------
If mask_root is provided, the dataset will load motion difference maps from:
  {mask_root}/{seq_name}.npz

The .npz file should contain key 'masks' with shape (T, H, W) float32,
where T matches the number of frames in the video.

See generate_masks_npz.py for the precomputation script.

Usage
-----
from datasets.ard100 import ARD100Dataset

ds = ARD100Dataset(
    root      = '/projects/prjs2041/datasets/ARD100',
    split     = 'train',
    mask_root = '/projects/prjs2041/datasets/ARD100/masks_npz',  # optional
)
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path

from .base import BaseUAVDataset

# Official YOLOMG train/test split (from prepare_ard100.py)
TRAIN_SEQS = [
    'phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19', 'phantom20',
    'phantom28', 'phantom29', 'phantom30', 'phantom32', 'phantom36', 'phantom40',
    'phantom42', 'phantom43', 'phantom44', 'phantom46', 'phantom63', 'phantom65',
    'phantom66', 'phantom68', 'phantom70', 'phantom71', 'phantom74', 'phantom75',
    'phantom76', 'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
    'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89', 'phantom90',
    'phantom101', 'phantom103', 'phantom104', 'phantom105', 'phantom106', 'phantom107',
    'phantom108', 'phantom109', 'phantom111', 'phantom112', 'phantom114', 'phantom115',
    'phantom116', 'phantom117', 'phantom118', 'phantom120', 'phantom132', 'phantom137',
    'phantom138', 'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
    'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150',
]

TEST_SEQS = [
    'phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08', 'phantom22',
    'phantom39', 'phantom41', 'phantom45', 'phantom47', 'phantom50', 'phantom54',
    'phantom55', 'phantom56', 'phantom57', 'phantom58', 'phantom60', 'phantom61',
    'phantom64', 'phantom73', 'phantom79', 'phantom92', 'phantom93', 'phantom94',
    'phantom95', 'phantom97', 'phantom102', 'phantom110', 'phantom113', 'phantom119',
    'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144',
]


class ARD100Dataset(BaseUAVDataset):

    def __init__(self, root, split='train', mask_root=None,
                 transform=None, skip_unannotated=True):
        """
        Args:
            root              : path to ARD100 root directory
            split             : 'train' or 'test'
            mask_root         : path to folder containing per-sequence .npz masks
                                (None = no motion mask returned)
            transform         : optional transform applied to uint8 BGR numpy frame
            skip_unannotated  : skip frames that have no matching XML file
        """
        self.root             = Path(root)
        self.split            = split
        self.mask_root        = Path(mask_root) if mask_root else None
        self.skip_unannotated = skip_unannotated
        # Cache for VideoCapture handles and loaded .npz masks
        self._caps       = {}   # seq_name -> cv2.VideoCapture
        self._npz_cache  = {}   # seq_name -> np.ndarray (T, H, W)
        super().__init__(transform=transform)

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_index(self):
        sequences   = TRAIN_SEQS if self.split == 'train' else TEST_SEQS
        video_dir   = self.root / f'{self.split}_videos'
        ann_root    = self.root / 'annotations'

        for seq in sequences:
            video_path = video_dir / f'{seq}.mp4'
            if not video_path.exists():
                print(f'[WARN] Video not found: {video_path}')
                continue

            seq_ann_dir = ann_root / seq
            if not seq_ann_dir.exists():
                print(f'[WARN] Annotation dir not found: {seq_ann_dir}')
                continue

            # Discover all XML files to know how many frames are annotated
            xml_files = sorted(seq_ann_dir.glob(f'{seq}_*.xml'))
            if not xml_files:
                print(f'[WARN] No XML files in {seq_ann_dir}')
                continue

            for xml_path in xml_files:
                # Parse frame index from filename: phantom09_0001.xml -> 0
                stem = xml_path.stem                    # e.g. 'phantom09_0001'
                frame_str = stem.rsplit('_', 1)[-1]    # '0001'
                frame_idx = int(frame_str) - 1          # 0-based

                # Pre-parse bounding box from XML to avoid XML overhead per call
                box, img_w, img_h = self._parse_xml(xml_path)

                self._index.append({
                    'seq':       seq,
                    'video':     str(video_path),
                    'frame':     frame_idx,
                    'box':       box,       # [xmin, ymin, xmax, ymax] pixels or None
                    'img_w':     img_w,
                    'img_h':     img_h,
                    'split':     self.split,
                })

    @staticmethod
    def _parse_xml(xml_path):
        """Return (box, img_w, img_h) from a Pascal VOC XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size  = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)

            obj = root.find('object')
            if obj is None:
                return None, img_w, img_h

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            return [xmin, ymin, xmax, ymax], img_w, img_h

        except Exception as e:
            print(f'[WARN] Failed to parse {xml_path}: {e}')
            return None, 640, 512

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _get_cap(self, seq, video_path):
        if seq not in self._caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f'Cannot open video: {video_path}')
            self._caps[seq] = cap
        return self._caps[seq]

    def _load_frame(self, entry):
        cap = self._get_cap(entry['seq'], entry['video'])

        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current != entry['frame']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, entry['frame'])
        ret, img = cap.read()
        if not ret:
            img = np.zeros((entry['img_h'], entry['img_w'], 3), dtype=np.uint8)

        # Build YOLO label from VOC [xmin, ymin, xmax, ymax]
        box = entry['box']
        if box is not None:
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            if w > 0 and h > 0:
                xc, yc, wn, hn = self.xywh_to_yolo(xmin, ymin, w, h,
                                                    entry['img_w'],
                                                    entry['img_h'])
                labels = np.array([[0, xc, yc, wn, hn]], dtype=np.float32)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        # Load precomputed motion mask if available
        mask = None
        if self.mask_root is not None:
            mask = self._load_mask(entry['seq'], entry['frame'],
                                   entry['img_h'], entry['img_w'])

        return img, labels, mask

    def _load_mask(self, seq, frame_idx, img_h, img_w):
        """Load one frame's motion mask from the .npz archive."""
        if seq not in self._npz_cache:
            npz_path = self.mask_root / f'{seq}.npz'
            if not npz_path.exists():
                return None
            self._npz_cache[seq] = np.load(str(npz_path))['masks']  # (T, H, W)

        masks = self._npz_cache[seq]
        if frame_idx >= len(masks):
            return None
        return masks[frame_idx].astype(np.float32)

    def __del__(self):
        for cap in self._caps.values():
            cap.release()
