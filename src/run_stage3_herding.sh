#!/bin/bash
#SBATCH --job-name=s3_herding
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/prjs2041/logs/stage3_herding_%j.out
#SBATCH --error=/projects/prjs2041/logs/stage3_herding_%j.err

set -euo pipefail

TORCHRUN=/home/knguyen1/.conda/envs/uav_master/bin/torchrun
SCRIPT=/projects/prjs2041/uav_code/train_stage3.py

STAGE2_WEIGHTS=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
HERDING_BUFFER=/projects/prjs2041/runs/stage2/seed42/herding_buffer.pt

# Read directly from project storage — avoids scratch quota collision
CST_ROOT=/projects/prjs2041/datasets/CST-AntiUAV
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

echo "=============================="
echo "Stage 3 — SCALE-STRATIFIED HERDING"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPUs    : $CUDA_VISIBLE_DEVICES"
echo "Started : $(date)"
echo "=============================="

if [ ! -f "$HERDING_BUFFER" ]; then
    echo "ERROR: Herding buffer not found at $HERDING_BUFFER"
    exit 1
fi
echo "Herding buffer : $HERDING_BUFFER"

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

echo "Using datasets from project storage (no local copy)"

$TORCHRUN \
    --standalone \
    --nproc_per_node=4 \
    $SCRIPT \
        --weights        $STAGE2_WEIGHTS \
        --replay-mode    herding \
        --replay-buffer  $HERDING_BUFFER \
        --replay-weight  4.0 \
        --name           herding \
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
echo "Results: /projects/prjs2041/runs/stage3/herding*/"
echo "=============================="
