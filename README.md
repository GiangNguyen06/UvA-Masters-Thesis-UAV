# UvA MSc Information Studies Thesis — UAV Detection with Continual Learning

**Author:** Khac Duc Giang Nguyen
**Institution:** University of Amsterdam, MSc Information Studies

## Overview

This repository contains all code, SLURM job scripts, dataset wrappers, and progress logs
for my MSc thesis on cross-domain continual learning for UAV detection.

The core idea: train a dual-stream IR+motion detector (YOLOMG) on a large labelled dataset,
then transfer it to a new domain without catastrophic forgetting, and measure how much it
forgets using a Forgetting Measure (FM = mAP_T1_after_T2 - mAP_T1_after_T1).

## Three-stage curriculum

| Stage | Task | Dataset | Status |
|-------|------|---------|--------|
| 1 | Full supervision — establish T1 mAP ceiling | Anti-UAV-RGBT (149K/62K frames) | Done — mAP@0.5 = 0.6617 |
| 2 | Teacher-Student UDA — cross-modal transfer | Anti-UAV 410 | Next |
| 3 | Continual fine-tuning (Scale-Stratified Herding) | CST | Planned |

## Model: YOLOMG

Dual-input YOLOv5-based detector (Guo et al., 2025):
- img1: IR appearance frame
- img2: Motion mask via Concat3 layer
- 318 layers, ~3M parameters, 640x640 input

At Stage 1, img2 is zeros (Anti-UAV-RGBT has no precomputed motion masks).
Model config: YOLOMG-main/models/dual_uav2.yaml

## Stage 1 Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.6617 |
| mAP@0.5:0.95 | 0.2900 |
| Best epoch | 54 / 99 |
| Training time | ~40 h (4x A100, Snellius HPC) |

mAP@0.5:0.95 is lower than mAP@0.5, which is expected for small UAV targets —
strict IoU thresholds are hard to satisfy when objects are only a few pixels wide.

## Repository Structure

    YOLOMG-main/              YOLOMG model source (Guo et al., 2025)
    src/
      datasets/               Dataset wrappers (Anti-UAV-RGBT, ARD100, CST, Anti-UAV410)
      train_stage1.py         Stage 1 DDP training (4x A100)
      train_stage2.py         Stage 2 training
      eval_stage1.py          Standalone eval — proper 10-threshold mAP@0.5:0.95
      run_stage1_ddp.sh       SLURM job: Stage 1
      run_eval_stage1.sh      SLURM job: standalone eval
      meeting_2.tex           Supervisor meeting log with full job history
    logs/                     All SLURM output logs (21 jobs, weeks 17-18)

## Installation (Snellius HPC)

    conda create -n uav_master python=3.9
    conda activate uav_master
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python-headless numpy scipy matplotlib pyyaml tqdm

## Running Stage 1

    sbatch src/run_stage1_ddp.sh
    sbatch src/run_eval_stage1.sh   # after training — computes real mAP@0.5:0.95

## Key Implementation Notes

**DDP:** torchrun --standalone --nproc_per_node=4, NCCL backend, 2h timeout.

**Distributed validation:** all 4 ranks validate in parallel via DistributedSampler +
all_gather_object, cutting val time from ~40 min to ~10 min.

**ap_per_class (YOLOMG-specific):** returns 7 values (not 5 like standard YOLOv5) and
requires a 2D correct array of shape (N, num_iou_thresholds). Training uses (N,1),
standalone eval uses (N,10) for accurate mAP@0.5:0.95.

**Workers:** use persistent_workers=False. With True, VideoCapture workers accumulate
CPU RAM over ~40h and trigger OOM. Stage 1 job 22522864 was killed at epoch 83
for this reason; best checkpoint was already saved at epoch 54.

**Dataset copying:** copy mp4 files to GPFS scratch before training (~73s for 5.1GB).
Do NOT pre-extract JPEG frames — same speed as VideoCapture but costs 4h to copy 211K files.
