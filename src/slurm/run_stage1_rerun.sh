#!/bin/bash
# =============================================================================
# run_stage1_rerun.sh
# =============================================================================
# Re-run Stage 1 with the updated training script that logs:
#   precision, recall, F1, mAP@0.5, mAP@0.5:0.95  per epoch
#   per-epoch checkpoints every 5 epochs
#
# Uses 4× A100 DDP (same as the original run).
# Will NOT overwrite existing runs — increment_path auto-creates a new dir:
#   /projects/prjs2041/runs/stage1/antiuav_rgbt15/  (or next available number)
#
# Submit with:
#   sbatch run_stage1_rerun.sh
#
# After it finishes, point Stage 2 at the new best.pt:
#   --weights /projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt
# =============================================================================
#SBATCH --job-name=s1_rerun
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:4
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/projects/prjs2041/logs/stage1_rerun_%j.out
#SBATCH --error=/projects/prjs2041/logs/stage1_rerun_%j.err

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
TORCHRUN=/home/knguyen1/.conda/envs/uav_master/bin/torchrun
SCRIPT=/projects/prjs2041/uav_code/train_stage1.py
SRC_DATA=/projects/prjs2041/datasets/Anti-UAV-RGBT

echo "========================================"
echo "Stage 1 Re-run (with P/R/F1 logging)"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPUs    : $CUDA_VISIBLE_DEVICES"
echo "Started : $(date)"
echo "========================================"

# ── Load environment ──────────────────────────────────────────────────────────
module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# ── Copy dataset to local SSD ($TMPDIR is fast NVMe on Snellius) ──────────────
echo ""
echo "Copying Anti-UAV-RGBT to \$TMPDIR ..."
T0=$(date +%s)
rsync -a --no-relative $SRC_DATA/ $TMPDIR/Anti-UAV-RGBT/
echo "Done in $(($(date +%s) - T0))s  |  $(du -sh $TMPDIR/Anti-UAV-RGBT | cut -f1)"

# ── Train ─────────────────────────────────────────────────────────────────────
echo ""
echo "Starting Stage 1 DDP training ..."
$TORCHRUN \
    --standalone \
    --nproc_per_node=4 \
    $SCRIPT \
        --epochs              100 \
        --batch-size          16 \
        --imgsz               640 \
        --workers             4 \
        --seed                42 \
        --dataset-root        $TMPDIR/Anti-UAV-RGBT \
        --name                antiuav_rgbt \
        --save-epoch-ckpts \
        --save-epoch-interval 5

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "Results in : /projects/prjs2041/runs/stage1/"
echo "========================================"
