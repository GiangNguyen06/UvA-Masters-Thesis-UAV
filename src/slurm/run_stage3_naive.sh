#!/bin/bash
#SBATCH --job-name=s3_naive
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/prjs2041/logs/stage3_naive_%j.out
#SBATCH --error=/projects/prjs2041/logs/stage3_naive_%j.err

set -euo pipefail

TORCHRUN=/home/knguyen1/.conda/envs/uav_master/bin/torchrun
SCRIPT=/projects/prjs2041/uav_code/train_stage3.py
STAGE2_WEIGHTS=/projects/prjs2041/runs/stage2/seed42/weights/best.pt

# Read directly from project storage — avoids scratch quota collision
# when multiple jobs share the same node.
CST_ROOT=/projects/prjs2041/datasets/CST-AntiUAV
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

echo "=============================="
echo "Stage 3 — NAIVE BASELINE"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPUs    : $CUDA_VISIBLE_DEVICES"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

echo "Using datasets from project storage (no local copy)"
echo "  CST  : $CST_ROOT"
echo "  RGBT : $RGBT_ROOT"

$TORCHRUN \
    --standalone \
    --nproc_per_node=4 \
    $SCRIPT \
        --weights        $STAGE2_WEIGHTS \
        --replay-mode    none \
        --name           naive \
        --epochs         50 \
        --batch-size     16 \
        --workers        4 \
        --seed           42 \
        --cst-root       $CST_ROOT \
        --rgbt-root      $RGBT_ROOT \
        --save-epoch-ckpts \
        --save-epoch-interval 5

echo "=============================="
echo "Done: $(date)"
echo "Results: /projects/prjs2041/runs/stage3/naive*/"
echo "=============================="
