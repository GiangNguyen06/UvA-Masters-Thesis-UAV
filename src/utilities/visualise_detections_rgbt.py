#!/usr/bin/env python3
"""
visualise_detections_rgbt.py
-----------------------------
Run a Stage 1 (or later) YOLOMG checkpoint on Anti-UAV-RGBT val frames
and save annotated images showing predicted bounding boxes vs ground truth.

Anti-UAV-RGBT layout:
  {root}/val/{sequence}/infrared.mp4   + infrared.json

Outputs:
  <out-dir>/frames/   annotated JPEG frames (pred=green, GT=orange)
  <out-dir>/video.mp4 assembled video per sequence (requires ffmpeg)
  <out-dir>/summary.txt  precision/recall per sequence

Usage (Snellius):
  python visualise_detections_rgbt.py \
      --weights  /projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt \
      --data-root $TMPDIR/Anti-UAV-RGBT \
      --out-dir   /projects/prjs2041/runs/stage1/antiuav_rgbt15/vis \
      --n-seqs    5 \
      --n-frames  60
"""

import sys
import os
import json
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

# ── Visual config ──────────────────────────────────────────────────────────────
IMG_W      = 640
IMG_H      = 512
IMG_SIZE   = 640
CONF_THRES = 0.25
IOU_THRES  = 0.45
NC         = 1

BOX_PRED   = (0, 200, 80)    # green  — prediction
BOX_GT     = (220, 80, 0)    # orange — ground truth
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS  = 2


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def letterbox_np(img_bgr: np.ndarray, size: int = 640):
    h, w  = img_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = size - nh, size - nw
    top, left    = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    return padded, scale, left, top


def unletterbox(x1, y1, x2, y2, scale, pad_l, pad_t, orig_w, orig_h):
    x1 = max(0, int((x1 - pad_l) / scale))
    y1 = max(0, int((y1 - pad_t) / scale))
    x2 = min(orig_w, int((x2 - pad_l) / scale))
    y2 = min(orig_h, int((y2 - pad_t) / scale))
    return x1, y1, x2, y2


def img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def draw_labeled_box(img, x1, y1, x2, y2, label, colour):
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    bg_y1 = max(0, y1 - th - 4)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 2),
                FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def iou_xywh(box_a, box_b):
    """IoU between two (x,y,w,h) boxes in pixel coords."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Sequence iteration  (reads directly from infrared.mp4)
# ══════════════════════════════════════════════════════════════════════════════

def iter_rgbt_sequences(data_root: Path, split: str = 'val',
                        n_seqs: int = 5, n_frames: int = 60):
    """
    Yield (seq_name, frame_idx, img_bgr, gt_box_or_None).
    gt_box: (x, y, w, h) in original pixel coords, or None if no UAV.
    """
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f'Split not found: {split_dir}')

    sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())[:n_seqs]
    for seq_dir in sequences:
        ann_path = seq_dir / 'infrared.json'
        video_path = seq_dir / 'infrared.mp4'
        if not ann_path.exists() or not video_path.exists():
            print(f'  [SKIP] {seq_dir.name} — missing infrared.json or infrared.mp4')
            continue

        with open(ann_path) as f:
            data = json.load(f)
        exist   = data.get('exist',   [])
        gt_rect = data.get('gt_rect', [])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f'  [SKIP] {seq_dir.name} — cannot open video')
            continue

        total = min(n_frames, len(exist))
        for frame_idx in range(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img_bgr = cap.read()
            if not ret:
                break
            e, box = exist[frame_idx], gt_rect[frame_idx]
            gt = tuple(int(v) for v in box) if e == 1 and box and len(box) == 4 else None
            yield seq_dir.name, frame_idx, img_bgr, gt

        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, img_bgr, device, imgsz, conf, iou):
    lb, scale, pad_l, pad_t = letterbox_np(img_bgr, imgsz)
    img1 = img_to_tensor(lb).unsqueeze(0).to(device)
    img2 = torch.zeros_like(img1)          # no motion mask at Stage 1

    out = model(img1, img2)
    if isinstance(out, tuple):
        out = out[0]
    dets = non_max_suppression(out, conf, iou)[0]

    boxes = []
    if dets is not None and len(dets):
        for *xyxy, conf_v, cls in dets.cpu().numpy():
            boxes.append((int(xyxy[0]), int(xyxy[1]),
                          int(xyxy[2]), int(xyxy[3]),
                          float(conf_v), int(cls)))
    return boxes, scale, pad_l, pad_t


def annotate(img_bgr, boxes, gt_box, scale, pad_l, pad_t):
    vis  = img_bgr.copy()
    orig_h, orig_w = img_bgr.shape[:2]

    # Ground truth (orange)
    if gt_box is not None:
        gx, gy, gw, gh = gt_box
        cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), BOX_GT, THICKNESS)
        cv2.putText(vis, 'GT', (gx, max(0, gy - 4)),
                    FONT, FONT_SCALE, BOX_GT, 1, cv2.LINE_AA)

    # Predictions (green)
    for x1, y1, x2, y2, conf_v, _ in boxes:
        ox1, oy1, ox2, oy2 = unletterbox(x1, y1, x2, y2, scale, pad_l, pad_t,
                                          orig_w, orig_h)
        draw_labeled_box(vis, ox1, oy1, ox2, oy2, f'UAV {conf_v:.2f}', BOX_PRED)

    cv2.putText(vis, f'preds:{len(boxes)}', (6, orig_h - 6),
                FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',   type=str,
        default='/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt')
    p.add_argument('--data-root', type=str,
        default='/gpfs/scratch1/shared/knguyen1/Anti-UAV-RGBT')
    p.add_argument('--out-dir',   type=str, default=None)
    p.add_argument('--split',     type=str, default='val')
    p.add_argument('--n-seqs',    type=int, default=5)
    p.add_argument('--n-frames',  type=int, default=60)
    p.add_argument('--conf',      type=float, default=CONF_THRES)
    p.add_argument('--iou',       type=float, default=IOU_THRES)
    p.add_argument('--device',    type=str, default='0')
    p.add_argument('--no-video',  action='store_true')
    return p.parse_args()


def assemble_video(frame_paths, out_path: Path, fps: int = 10):
    list_path = out_path.parent / '_framelist.txt'
    with open(list_path, 'w') as f:
        for p in sorted(frame_paths):
            f.write(f"file '{p}'\nduration {1/fps:.4f}\n")
    ret = os.system(
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
        f'-vcodec libx264 -pix_fmt yuv420p -crf 23 "{out_path}" -loglevel warning'
    )
    list_path.unlink(missing_ok=True)
    if ret == 0:
        print(f'  Video: {out_path}')
    else:
        print(f'  [WARN] ffmpeg failed (code {ret})')


def main():
    args   = parse_args()
    device = select_device(args.device)

    weights   = Path(args.weights)
    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir) if args.out_dir \
                else weights.parent.parent / 'vis_rgbt'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading: {weights}')
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = (ckpt['ema'] if ckpt.get('ema') else ckpt['model']).float().to(device)
    model.nc = NC; model.names = ['UAV']; model.eval()
    print(f'  Epoch {ckpt.get("epoch","?")}  |  '
          f'best_fitness={ckpt.get("best_fitness","?")}\n')

    stats      = {}
    all_frames = []

    print(f'Visualising {args.n_seqs} seqs × {args.n_frames} frames '
          f'from {data_root}/{args.split}')

    for seq, fidx, img_bgr, gt_box in iter_rgbt_sequences(
            data_root, args.split, args.n_seqs, args.n_frames):

        if seq not in stats:
            stats[seq] = {'tp': 0, 'fp': 0, 'fn': 0, 'frames': 0}
            print(f'  → {seq}')

        boxes, scale, pad_l, pad_t = run_inference(
            model, img_bgr, device, IMG_SIZE, args.conf, args.iou)

        vis = annotate(img_bgr, boxes, gt_box, scale, pad_l, pad_t)

        out_path = frames_dir / f'{seq}_{fidx:04d}.jpg'
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        all_frames.append(str(out_path))

        # Simple TP/FP/FN at IoU ≥ 0.5
        s = stats[seq]; s['frames'] += 1
        orig_h, orig_w = img_bgr.shape[:2]
        if gt_box is not None:
            matched = False
            for x1, y1, x2, y2, *_ in boxes:
                ox1, oy1, ox2, oy2 = unletterbox(
                    x1, y1, x2, y2, scale, pad_l, pad_t, orig_w, orig_h)
                pred_xywh = (ox1, oy1, ox2 - ox1, oy2 - oy1)
                if iou_xywh(gt_box, pred_xywh) >= 0.5:
                    matched = True; break
            if matched:  s['tp'] += 1
            else:        s['fn'] += 1
            if not matched and boxes:
                s['fp'] += len(boxes)
        elif boxes:
            s['fp'] += len(boxes)

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = [
        'Stage 1 Detection Visualisation — Anti-UAV-RGBT val',
        f'Checkpoint : {weights}',
        f'conf={args.conf}  iou={args.iou}',
        '',
        f'{"Seq":<28} {"Fr":>5} {"TP":>5} {"FP":>5} {"FN":>5} {"P":>6} {"R":>6}',
        '-' * 62,
    ]
    tot_tp = tot_fp = tot_fn = 0
    for seq, s in sorted(stats.items()):
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        p = tp / (tp + fp) if tp + fp else float('nan')
        r = tp / (tp + fn) if tp + fn else float('nan')
        lines.append(f'{seq:<28} {s["frames"]:>5} {tp:>5} {fp:>5} {fn:>5} '
                      f'{p:>6.3f} {r:>6.3f}')
        tot_tp += tp; tot_fp += fp; tot_fn += fn

    p_all = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else float('nan')
    r_all = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else float('nan')
    total_frames = sum(s['frames'] for s in stats.values())
    lines += ['-' * 62,
              f'{"TOTAL":<28} {total_frames:>5} {tot_tp:>5} {tot_fp:>5} {tot_fn:>5} '
              f'{p_all:>6.3f} {r_all:>6.3f}']

    txt = '\n'.join(lines)
    print('\n' + txt)
    (out_dir / 'summary.txt').write_text(txt)

    if not args.no_video and all_frames:
        assemble_video(all_frames, out_dir / 'detections_rgbt.mp4')

    print(f'\n{len(all_frames)} frames → {frames_dir}')


if __name__ == '__main__':
    main()
