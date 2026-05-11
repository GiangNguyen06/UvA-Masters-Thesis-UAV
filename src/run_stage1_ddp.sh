#!/bin/bash
# =============================================================
# run_stage1_ddp.sh
# Stage 1: Train YOLOMG on Anti-UAV-RGBT — 4× A100 with DDP
#
# Why DDP instead of DataParallel:
#   DataParallel: scatter/gather all activations over PCIe → 3× SLOWER
#   DDP:          all-reduce only gradients over NVLink    → ~4× FASTER
#
# Each GPU processes BS=16 samples independently.
# DDP all-reduces gradients across 4 GPUs → equivalent to BS=64
# but with 4× fewer batches per GPU per epoch.
#
# Expected timing:
#   ~9346 / 4 = 2337 batches/epoch/GPU
#   2337 × 0.64 s/batch = 1496 s ≈ 25 min/epoch
#   100 epochs ≈ 41 h  (well within 120 h limit)
#
# Submit with:
#   sbatch run_stage1_ddp.sh
#
# Outputs:
#   /projects/prjs2041/runs/stage1/antiuav_rgbt*/
#       weights/best.pt       ← use as --weights for Stage 2
#       weights/last.pt
#       results.csv
#       stage1_mAP.txt        ← T1 mAP for Forgetting Measure
#   /projects/prjs2041/logs/stage1_ddp_<JOBID>.out
# =============================================================
#SBATCH --job-name=stage1_ddp
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:4
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/projects/prjs2041/logs/stage1_ddp_%j.out

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
TORCHRUN=/home/knguyen1/.conda/envs/uav_master/bin/torchrun
SCRIPT=/projects/prjs2041/uav_code/train_stage1.py
EXTRACT_SCRIPT=/projects/prjs2041/uav_code/extract_frames_antiuav_rgbt.py
LOGS=/projects/prjs2041/logs
SRC_DATA=/projects/prjs2041/datasets/Anti-UAV-RGBT   # JSON + mp4 files

LOCAL_SCRATCH=/scratch-local/${USER}.${SLURM_JOB_ID}
LOCAL_DATA=${LOCAL_SCRATCH}/Anti-UAV-RGBT

mkdir -p $LOGS

echo "========================================"
echo "Stage 1 DDP: YOLOMG on Anti-UAV-RGBT"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo "Time:    $(date)"
echo "========================================"
echo ""

echo "Local scratch:"
df -h $LOCAL_SCRATCH
echo ""

# ── Step 1: Copy dataset (JSON + mp4) to GPFS scratch (~73 s, 227 files) ──────
echo "Copying Anti-UAV-RGBT to GPFS scratch..."
T0=$(date +%s)
cp -r $SRC_DATA $LOCAL_DATA
echo "Done in $(($(date +%s) - T0))s.  Size: $(du -sh $LOCAL_DATA | cut -f1)"
echo ""

# ── Step 3: Train with torchrun (4 processes, one per GPU) ────────────────────
module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# torchrun --standalone handles master_addr/port automatically for single-node
$TORCHRUN \
    --standalone \
    --nproc_per_node=4 \
    $SCRIPT \
    --epochs       100 \
    --batch-size   16 \
    --imgsz        640 \
    --workers      4 \
    --dataset-root $LOCAL_DATA \
    --name         antiuav_rgbt

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "Check results: /projects/prjs2041/runs/stage1/"
echo "========================================"
