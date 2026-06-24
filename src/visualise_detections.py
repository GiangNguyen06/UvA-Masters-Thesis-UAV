#!/usr/bin/env python3
"""
visualise_detections.py
-----------------------
Run Stage 2 YOLOMG checkpoint on Anti-UAV410 val frames and save
annotated images showing predicted bounding boxes with confidence scores.
Also assembles an MP4 video from the annotated frames.

Requires:
  - Stage 2 best.pt checkpoint
  - Anti-UAV410 dataset on scratch (copy before running)
  - Motion masks .npz file (or zeros fallback if not available)

Usage (Snellius interactive / sbatch):
  python visualise_detections.py \
      --weights  /projects/prjs2041/runs/stage2/<run>/weights/best.pt \
      --data-root $TMPDIR/Anti-UAV410 \
      --masks-root $TMPDIR/masks_antiuav410 \
      --out-dir   /projects/prjs2041/runs/stage2/<run>/vis \
      --n-seqs    5 \
      --n-frames  60

Outputs:
  <out-dir>/frames/   annotated JPEG frames
  <out-dir>/video.mp4 assembled video (requires ffmpeg on PATH)
  <out-dir>/summary.txt  detection stats per sequence
"""

import sys
import os
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
YOLOMG_ROOT = Path('/projects/prjs2041/YOLOMG')
UAV_CODE    = Path('/projects/prjs2041/uav_code')

sys.path.insert(0, str(YOLOMG_ROOT))
sys.path.insert(0, str(UAV_CODE))

from models.yolo import Model
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE   = 640
CONF_THRES = 0.25   # higher than eval threshold — cleaner visualisation
IOU_THRES  = 0.45
NC         = 1

# Colour scheme: muted palette, Tufte-inspired
BOX_COL_PRED = (0, 200, 80)    # BGR green  — predictions
BOX_COL_GT   = (220, 80, 0)    # BGR orange — ground truth
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.55
THICKNESS    = 2


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def letterbox_np(img: np.ndarray, size: int = 640):
    """
    Letterbox a uint8 BGR image to (size, size).
    Returns (padded_img, scale, (pad_left, pad_top)).
    """
    h, w   = img.shape[:2]
    scale  = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r  = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = size - nh, size - nw
    top,  left   = pad_h // 2, pad_w // 2
    pad  = cv2.copyMakeBorder(
        img_r, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    return pad, scale, (left, top)


def unletterbox_box(x1, y1, x2, y2, scale, pad_left, pad_top, orig_w, orig_h):
    """Map letterboxed pixel coords back to original image coords."""
    x1 = max(0, (x1 - pad_left) / scale)
    y1 = max(0, (y1 - pad_top)  / scale)
    x2 = min(orig_w, (x2 - pad_left) / scale)
    y2 = min(orig_h, (y2 - pad_top)  / scale)
    return int(x1), int(y1), int(x2), int(y2)


def img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """uint8 BGR HWC → float32 RGB CHW [0, 1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t


def load_mask(masks_root: Path, seq: str, frame_idx: int, size: int) -> torch.Tensor:
    """
    Load motion residual mask for this frame.
    Tries <masks_root>/<seq>.npz first; falls back to zero tensor.
    The .npz is expected to store masks under key 'masks' with shape (N, H, W).
    """
    if masks_root is not None:
        npz_path = masks_root / f'{seq}.npz'
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                masks = data['masks']           # (N, H, W) float32 [0, 1]
                if frame_idx < len(masks):
                    m = masks[frame_idx]        # (H, W)
                    m = np.stack([m, m, m], axis=0)  # (3, H, W) to match img
                    t = torch.from_numpy(m).float()
                    # Letterbox to match img1 size
                    t = F.interpolate(
                        t.unsqueeze(0), size=(size, size),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                    return t
            except Exception as e:
                print(f'  [WARN] mask load failed for {seq}/{frame_idx}: {e}')

    # Fallback: zero tensor (model still runs, motion branch is inactive)
    return torch.zeros(3, size, size)


def draw_box(img, x1, y1, x2, y2, label: str, colour):
    """Draw a labelled bounding box onto img (in-place)."""
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    bg_y1 = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 2),
                FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset iteration (no Dataset class — keep it simple for vis)
# ══════════════════════════════════════════════════════════════════════════════

import json

def iter_sequences(data_root: Path, split: str = 'val',
                   n_seqs: int = 5, n_frames: int = 60):
    """
    Yield (seq_name, frame_idx, img_bgr, gt_box_or_None) for the
    first n_frames of the first n_seqs sequences in the val split.
    gt_box: (x, y, w, h) in pixel coords, or None if UAV absent.
    """
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f'Split not found: {split_dir}')

    sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())[:n_seqs]
    for seq_dir in sequences:
        ann_path = seq_dir / 'IR_label.json'
        if not ann_path.exists():
            print(f'  [WARN] no IR_label.json in {seq_dir.name}, skipping.')
            continue

        with open(ann_path) as f:
            data = json.load(f)

        exist   = data.get('exist',   [])
        gt_rect = data.get('gt_rect', [])

        for frame_idx in range(min(n_frames, len(exist))):
            e, box = exist[frame_idx], gt_rect[frame_idx]

            img_name = f'{seq_dir.name}_{frame_idx + 1:04d}.jpg'
            img_path = seq_dir / img_name
            if not img_path.exists():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            gt = tuple(box) if e == 1 and box and len(box) == 4 else None
            yield seq_dir.name, frame_idx, img_bgr, gt


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualise Stage 2 detections on Anti-UAV410 val set.'
    )
    p.add_argument('--weights',    type=str,
                   default='/projects/prjs2041/runs/stage2/stage2_uda1/weights/best.pt',
                   help='Path to Stage 2 best.pt checkpoint')
    p.add_argument('--data-root',  type=str,
                   default='/gpfs/scratch1/shared/knguyen1/Anti-UAV410',
                   help='Anti-UAV410 root (copy to $TMPDIR before running)')
    p.add_argument('--masks-root', type=str, default=None,
                   help='Directory of per-sequence .npz motion masks '
                        '(optional; zeros used if absent)')
    p.add_argument('--out-dir',    type=str, default=None,
                   help='Output directory (default: weights_dir/vis/)')
    p.add_argument('--split',      type=str, default='val')
    p.add_argument('--n-seqs',     type=int, default=5,
                   help='Number of sequences to visualise')
    p.add_argument('--n-frames',   type=int, default=60,
                   help='Max frames per sequence')
    p.add_argument('--conf',       type=float, default=CONF_THRES)
    p.add_argument('--iou',        type=float, default=IOU_THRES)
    p.add_argument('--device',     type=str, default='0')
    p.add_argument('--no-video',   action='store_true',
                   help='Skip MP4 assembly (saves time if ffmpeg unavailable)')
    return p.parse_args()


@torch.no_grad()
def run_inference(model, img_bgr: np.ndarray, img2_t: torch.Tensor,
                  device, imgsz: int, conf: float, iou: float):
    """
    Returns list of detections: [(x1,y1,x2,y2,conf,cls), ...] in letterboxed coords.
    Also returns (scale, pad_left, pad_top) for unletterboxing.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    lb, scale, (pad_left, pad_top) = letterbox_np(img_bgr, imgsz)

    img1_t = img_to_tensor(lb).to(device).unsqueeze(0)   # (1, 3, 640, 640)
    img2_t = img2_t.to(device).unsqueeze(0)

    preds = model(img1_t, img2_t)
    if isinstance(preds, tuple):
        preds = preds[0]
    dets = non_max_suppression(preds, conf, iou)[0]   # (N, 6) or empty

    boxes = []
    if dets is not None and len(dets):
        for *xyxy, conf_val, cls in dets.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            boxes.append((x1, y1, x2, y2, float(conf_val), int(cls)))

    return boxes, scale, pad_left, pad_top


def annotate_frame(img_bgr: np.ndarray, boxes, gt_box,
                   scale, pad_left, pad_top):
    """
    Draw predicted bboxes (green) and GT bbox (orange) on a copy of img_bgr.
    boxes: [(x1,y1,x2,y2,conf,cls), ...] in letterboxed coords.
    gt_box: (x, y, w, h) in original pixel coords, or None.
    """
    vis = img_bgr.copy()
    orig_h, orig_w = img_bgr.shape[:2]

    # Draw ground truth (orange)
    if gt_box is not None:
        gx, gy, gw, gh = [int(v) for v in gt_box]
        cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), BOX_COL_GT, THICKNESS)
        cv2.putText(vis, 'GT', (gx, max(0, gy - 4)),
                    FONT, FONT_SCALE, BOX_COL_GT, 1, cv2.LINE_AA)

    # Draw predictions (green)
    for x1, y1, x2, y2, conf_val, cls in boxes:
        ox1, oy1, ox2, oy2 = unletterbox_box(
            x1, y1, x2, y2, scale, pad_left, pad_top, orig_w, orig_h
        )
        label = f'UAV {conf_val:.2f}'
        draw_box(vis, ox1, oy1, ox2, oy2, label, BOX_COL_PRED)

    # Overlay: n_pred count
    n_pred = len(boxes)
    cv2.putText(vis, f'preds: {n_pred}', (8, orig_h - 8),
                FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def assemble_video(frame_paths, out_path: Path, fps: int = 10):
    """Assemble sorted list of frame paths into an MP4 using ffmpeg."""
    # Write a file list for ffmpeg concat demuxer
    list_path = out_path.parent / '_framelist.txt'
    with open(list_path, 'w') as f:
        for p in frame_paths:
            f.write(f"file '{p}'\nduration {1/fps:.4f}\n")

    cmd = (
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
        f'-vcodec libx264 -pix_fmt yuv420p -crf 23 "{out_path}" '
        f'-loglevel warning'
    )
    ret = os.system(cmd)
    list_path.unlink(missing_ok=True)
    if ret == 0:
        print(f'  Video saved: {out_path}')
    else:
        print(f'  [WARN] ffmpeg failed (return code {ret}). '
              f'Frames are in {out_path.parent}/frames/')


def main():
    args = parse_args()
    device = select_device(args.device)

    weights    = Path(args.weights)
    data_root  = Path(args.data_root)
    masks_root = Path(args.masks_root) if args.masks_root else None
    out_dir    = Path(args.out_dir) if args.out_dir \
                 else weights.parent.parent / 'vis'

    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {weights}')
    ckpt = torch.load(weights, map_location=device, weights_only=False)

    if 'ema' in ckpt and ckpt['ema'] is not None:
        model = ckpt['ema'].float().to(device)
        print('  Using EMA weights')
    else:
        model = ckpt['model'].float().to(device)
        print('  Using model weights (no EMA)')

    model.nc    = NC
    model.names = ['UAV']
    model.eval()

    epoch = ckpt.get('epoch', '?')
    print(f'  Checkpoint epoch : {epoch}')
    print(f'  Reported best_t2 : {ckpt.get("best_t2", "?")}\n')

    # ── Run inference and annotate ────────────────────────────────────────────
    stats         = {}   # seq → {tp, fp, fn}
    all_frame_paths = []

    print(f'Visualising {args.n_seqs} sequences × {args.n_frames} frames …')
    print(f'Output dir: {out_dir}\n')

    for seq, frame_idx, img_bgr, gt_box in iter_sequences(
        data_root, args.split, args.n_seqs, args.n_frames
    ):
        if seq not in stats:
            stats[seq] = {'tp': 0, 'fp': 0, 'fn': 0, 'frames': 0}
            print(f'  Sequence: {seq}')

        # Load motion mask (zeros if unavailable)
        img2_t = load_mask(masks_root, seq, frame_idx, IMG_SIZE)

        # Inference
        boxes, scale, pad_left, pad_top = run_inference(
            model, img_bgr, img2_t, device,
            IMG_SIZE, args.conf, args.iou
        )

        # Annotate frame
        vis = annotate_frame(img_bgr, boxes, gt_box, scale, pad_left, pad_top)

        # Save
        frame_name = f'{seq}_{frame_idx:04d}.jpg'
        out_path   = frames_dir / frame_name
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        all_frame_paths.append(str(out_path))

        # Simple TP/FP/FN tracking (IoU > 0.5 against GT)
        s = stats[seq]
        s['frames'] += 1
        if gt_box is not None:
            gx, gy, gw, gh = gt_box
            gt_xyxy = np.array([[gx, gy, gx + gw, gy + gh]], dtype=np.float32)
            matched = False
            if boxes:
                orig_h, orig_w = img_bgr.shape[:2]
                for x1, y1, x2, y2, *_ in boxes:
                    ox1, oy1, ox2, oy2 = unletterbox_box(
                        x1, y1, x2, y2, scale, pad_left, pad_top, orig_w, orig_h
                    )
                    pred_xyxy = np.array([[ox1, oy1, ox2, oy2]], dtype=np.float32)
                    # Compute IoU
                    ix1 = max(gt_xyxy[0,0], pred_xyxy[0,0])
                    iy1 = max(gt_xyxy[0,1], pred_xyxy[0,1])
                    ix2 = min(gt_xyxy[0,2], pred_xyxy[0,2])
                    iy2 = min(gt_xyxy[0,3], pred_xyxy[0,3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_gt   = gw * gh
                    area_pred = (ox2 - ox1) * (oy2 - oy1)
                    union = area_gt + area_pred - inter
                    if union > 0 and inter / union >= 0.5:
                        matched = True
                        break
            if matched:
                s['tp'] += 1
            else:
                s['fn'] += 1
            if not matched and boxes:
                s['fp'] += len(boxes)
        elif boxes:
            s['fp'] += len(boxes)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_lines = [
        f'Stage 2 Detection Visualisation Summary',
        f'Checkpoint : {weights}',
        f'Split      : {args.split}',
        f'conf_thres : {args.conf}  |  iou_thres : {args.iou}',
        f'',
        f'{"Sequence":<30} {"Frames":>6} {"TP":>5} {"FP":>5} {"FN":>5} {"Prec":>6} {"Rec":>6}',
        '-' * 70,
    ]
    total_tp = total_fp = total_fn = 0
    for seq, s in sorted(stats.items()):
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        summary_lines.append(
            f'{seq:<30} {s["frames"]:>6} {tp:>5} {fp:>5} {fn:>5} '
            f'{prec:>6.3f} {rec:>6.3f}'
        )
        total_tp += tp; total_fp += fp; total_fn += fn

    p_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
    r_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
    summary_lines += [
        '-' * 70,
        f'{"TOTAL":<30} {sum(s["frames"] for s in stats.values()):>6} '
        f'{total_tp:>5} {total_fp:>5} {total_fn:>5} '
        f'{p_all:>6.3f} {r_all:>6.3f}',
    ]

    summary_text = '\n'.join(summary_lines)
    print('\n' + summary_text)
    (out_dir / 'summary.txt').write_text(summary_text)

    # ── Assemble video ────────────────────────────────────────────────────────
    if not args.no_video and all_frame_paths:
        video_path = out_dir / 'detections.mp4'
        assemble_video(sorted(all_frame_paths), video_path, fps=10)

    print(f'\nDone. {len(all_frame_paths)} frames saved to {frames_dir}')


if __name__ == '__main__':
    main()
