#!/usr/bin/env python3
"""
generate_masks_npz.py
---------------------
Precomputes FD5 motion difference masks for all ARD100 sequences and stores
them as one .npz file per sequence, avoiding per-frame file extraction.

Output
------
/projects/prjs2041/datasets/ARD100/masks_npz/
    phantom09.npz   ->  array 'masks' shape (T, H, W) uint8
    phantom10.npz
    ...

The mask for frame i is the FD5 motion difference computed from frames
[i-2, i, i+2] (every 2 frames). Frames 0-1 and T-2,T-1 have zero masks
because the sliding window needs 5 frames to produce a result.

Algorithm (inlined from FD5_mask.py, without disk writes)
----------------------------------------------------------
1. Gaussian blur each of the 3 frames
2. Convert to grayscale
3. ECC homography align frame[i-2] → frame[i]  (motion_compensate)
4. absdiff(frame[i], aligned[i-2])  → frameDiff1
5. ECC homography align frame[i+2] → frame[i]  (motion_compensate)
6. absdiff(frame[i], aligned[i+2])  → frameDiff2
7. mask = (frameDiff1 + frameDiff2) / 2

Usage
-----
Submit as SLURM job:
    sbatch run_generate_masks.sh

Or run directly (slow on login node, only for testing):
    /home/knguyen1/.conda/envs/uav_master/bin/python generate_masks_npz.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ── Inlined from MOD_Functions.py (removes onnxruntime dependency) ────────────

def motion_compensate(frame1, frame2):
    """
    Estimate homography between frame1 and frame2 using Lucas-Kanade optical
    flow on a sparse grid, then warp frame1 to align with frame2.

    Accepts both BGR (3-channel) and grayscale (2-D) uint8 arrays.

    Returns
    -------
    compensated : np.ndarray  warped frame1 aligned to frame2
    mask        : np.ndarray  uint8 validity mask (255 = valid region)
    avg_dst     : float       mean optical-flow distance
    motion_x    : float       mean x translation
    motion_y    : float       mean y translation
    homography_matrix : np.ndarray  (3, 3) float64
    """
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    # Work with grayscale for optical flow
    def to_gray(f):
        if f.ndim == 2:
            return f
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    width  = frame2.shape[1]
    height = frame2.shape[0]
    scale  = 2

    f1_gray = to_gray(frame1)
    f2_gray = to_gray(frame2)

    frame1_grid = cv2.resize(f1_gray, (960 * scale, 540 * scale),
                             interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(f2_gray, (960 * scale, 540 * scale),
                             interpolation=cv2.INTER_CUBIC)

    width_grid  = frame2_grid.shape[1]
    height_grid = frame2_grid.shape[0]
    gridSizeW = 32 * 2
    gridSizeH = 24 * 2

    p1 = []
    grid_numW = int(width_grid  / gridSizeW - 1)
    grid_numH = int(height_grid / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0),
                     np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1        = np.array(p1)
    pts_num   = grid_numW * grid_numH
    pts_prev  = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(
        frame1_grid, frame2_grid, pts_prev, None, **lk_params)

    good_new = pts_cur[st == 1]
    good_old = pts_prev[st == 1]

    motion_distance, translate_x, translate_y = [], [], []
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        dist0 = np.sqrt((a - c) ** 2 + (b - d) ** 2)
        if dist0 > 50:
            continue
        motion_distance.append(dist0)
        translate_x.append(a - c)
        translate_y.append(b - d)

    avg_dst  = np.mean(motion_distance) if motion_distance else 0.0
    motion_x = np.mean(translate_x)    if translate_x    else 0.0
    motion_y = np.mean(translate_y)    if translate_y    else 0.0

    if len(good_old) < 15:
        homography_matrix = np.array([[0.999, 0, 0],
                                      [0, 0.999, 0],
                                      [0, 0,     1]], dtype=np.float64)
    else:
        homography_matrix, _ = cv2.findHomography(good_new, good_old,
                                                  cv2.RANSAC, 3.0)

    compensated = cv2.warpPerspective(
        frame1, homography_matrix, (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                      dtype=np.float32).reshape(-1, 1, 2)
    homo_inv       = np.linalg.inv(homography_matrix)
    vertex_trans   = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)

    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask, avg_dst, motion_x, motion_y, homography_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path('/projects/prjs2041/datasets/ARD100')
TRAIN_VIDEOS = DATASET_ROOT / 'train_videos'
TEST_VIDEOS  = DATASET_ROOT / 'test_videos'
OUTPUT_DIR   = DATASET_ROOT / 'masks_npz'

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


# ── Core computation (inlined from FD5_mask.py) ───────────────────────────────

def compute_fd5(frame_old, frame_mid, frame_new):
    """
    Compute FD5 motion difference mask for the middle frame.

    Parameters
    ----------
    frame_old : np.ndarray  BGR uint8, frame at t-4
    frame_mid : np.ndarray  BGR uint8, frame at t-2 (mask is for this frame)
    frame_new : np.ndarray  BGR uint8, frame at t

    Returns
    -------
    mask : np.ndarray  float32 (H, W), values in [0, 255]
    """
    # Blur and convert to grayscale
    f_old = cv2.cvtColor(cv2.GaussianBlur(frame_old, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    f_mid = cv2.cvtColor(cv2.GaussianBlur(frame_mid, (11, 11), 0), cv2.COLOR_BGR2GRAY)
    f_new = cv2.cvtColor(cv2.GaussianBlur(frame_new, (11, 11), 0), cv2.COLOR_BGR2GRAY)

    # Align old → mid, compute difference
    img_comp1, _, _, _, _, _ = motion_compensate(f_old, f_mid)
    diff1 = cv2.absdiff(f_mid, img_comp1).astype(np.float32)

    # Align new → mid, compute difference
    img_comp2, _, _, _, _, _ = motion_compensate(f_new, f_mid)
    diff2 = cv2.absdiff(f_mid, img_comp2).astype(np.float32)

    return (diff1 + diff2) / 2.0


# ── Sequence processing ───────────────────────────────────────────────────────

def process_sequence(video_path: Path, out_path: Path):
    """
    Read a full video, compute FD5 masks for all frames, save as .npz.
    Skips if output already exists.
    """
    if out_path.exists():
        print(f'  [SKIP] Already exists: {out_path.name}')
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  [WARN] Cannot open: {video_path}')
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Pre-allocate output array (uint8 saves ~4x vs float32)
    masks = np.zeros((total_frames, h, w), dtype=np.uint8)

    # Read all frames into a sliding window buffer
    # Buffer holds last 5 frames: [t-4, t-3, t-2, t-1, t]
    buf = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buf.append(frame)
        if len(buf) > 5:
            buf.pop(0)

        # Once we have 5 frames, compute mask for the middle frame (buf[2])
        if len(buf) == 5:
            mid_idx = frame_idx - 2   # 0-based index of the middle frame
            try:
                mask = compute_fd5(buf[0], buf[2], buf[4])
                masks[mid_idx] = np.clip(mask, 0, 255).astype(np.uint8)
            except Exception as e:
                # motion_compensate can fail on degenerate frames — leave as zeros
                pass

        frame_idx += 1

    cap.release()

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), masks=masks)
    print(f'  Saved: {out_path.name}  ({total_frames} frames, '
          f'{out_path.stat().st_size / 1e6:.1f} MB)')
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('FD5 Motion Mask Generation → .npz')
    print('=' * 50)
    print(f'Output: {OUTPUT_DIR}')
    print()

    all_seqs = (
        [(s, TRAIN_VIDEOS / f'{s}.mp4') for s in TRAIN_SEQS] +
        [(s, TEST_VIDEOS  / f'{s}.mp4') for s in TEST_SEQS]
    )

    success, skipped, failed = 0, 0, 0

    for seq_name, video_path in tqdm(all_seqs, desc='Sequences', unit='seq'):
        out_path = OUTPUT_DIR / f'{seq_name}.npz'

        if out_path.exists():
            skipped += 1
            continue

        print(f'\n[{seq_name}]')
        if not video_path.exists():
            print(f'  [WARN] Video not found: {video_path}')
            failed += 1
            continue

        ok = process_sequence(video_path, out_path)
        if ok:
            success += 1
        else:
            failed += 1

    print('\n' + '=' * 50)
    print(f'Done.  Generated: {success}  Skipped: {skipped}  Failed: {failed}')
    print(f'Output directory: {OUTPUT_DIR}')
    npz_files = list(OUTPUT_DIR.glob('*.npz'))
    if npz_files:
        total_mb = sum(f.stat().st_size for f in npz_files) / 1e6
        print(f'Total .npz files: {len(npz_files)}  ({total_mb:.1f} MB)')


if __name__ == '__main__':
    main()
