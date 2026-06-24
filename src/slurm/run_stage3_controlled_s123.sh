#!/bin/bash
#SBATCH --job-name=s3_ctrl_s123
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/projects/prjs2041/logs/stage3_ctrl_s123_%j.out
#SBATCH --error=/projects/prjs2041/logs/stage3_ctrl_s123_%j.err

# Controlled Stage 3 rerun — seed 123, lr0=1e-3 (matches Stage 2).
# Capped at 10 epochs: best T1 collapse occurs at epoch 3 in the
# canonical run; 10 epochs is sufficient to confirm the pattern.
# Uses A100 (not H100) to stay within remaining SBU budget.

set -euo pipefail

TORCHRUN=/home/knguyen1/.conda/envs/uav_master/bin/torchrun
SCRIPT=/projects/prjs2041/uav_code/train_stage3.py
STAGE2_WEIGHTS=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
CST_ROOT=/projects/prjs2041/datasets/CST-AntiUAV
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

echo "=============================="
echo "Stage 3 — CONTROLLED (lr0=1e-3) — seed 123"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

$TORCHRUN \
    --standalone \
    --nproc_per_node=4 \
    $SCRIPT \
        --weights        $STAGE2_WEIGHTS \
        --replay-mode    none \
        --name           naive_ctrl_s123 \
        --epochs         8 \
        --batch-size     16 \
        --lr0            1e-3 \
        --workers        4 \
        --seed           123 \
        --cst-root       $CST_ROOT \
        --rgbt-root      $RGBT_ROOT \
        --save-epoch-ckpts \
        --save-epoch-interval 1

echo "=============================="
echo "Done: $(date)"
echo "=============================="
