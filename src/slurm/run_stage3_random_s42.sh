#!/bin/bash
#SBATCH --job-name=s3_rand_s42
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/projects/prjs2041/logs/stage3_random_s42_%j.out
#SBATCH --error=/projects/prjs2041/logs/stage3_random_s42_%j.err
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=giang.nguyen5@student.uva.nl

# Stage 3 — Random-Stratified replay, seed 42, epoch-capped at 3
# (reaches epoch 2, SSH's best-T3 checkpoint, for a like-for-like ablation).
# Ablation baseline against SSH: same buffer size and stratum proportions,
# random selection instead of herding. Distinguishes SSH's feature-space
# selection from the benefit of replay alone.
# Prerequisite: run_build_buffer_train.sh must complete first.

set -euo pipefail

TORCHRUN=/home/knguyen1/.conda/envs/uav_master/bin/torchrun
SCRIPT=/projects/prjs2041/uav_code/train_stage3.py

STAGE2_WEIGHTS=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
RANDOM_BUFFER=/projects/prjs2041/runs/stage2/seed42/herding_buffer_random_train.pt
CST_ROOT=/projects/prjs2041/datasets/CST-AntiUAV
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

echo "=============================="
echo "Stage 3 — Random-Stratified (train-split buffer) — seed 42"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

if [ ! -f "$RANDOM_BUFFER" ]; then
    echo "ERROR: herding_buffer_random_train.pt not found — run run_build_buffer_train.sh first"
    exit 1
fi

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

$TORCHRUN \
    --standalone \
    --nproc_per_node=4 \
    $SCRIPT \
        --weights        $STAGE2_WEIGHTS \
        --replay-mode    random_stratified \
        --replay-buffer  $RANDOM_BUFFER \
        --replay-weight  4.0 \
        --name           random_s42 \
        --epochs         3 \
        --batch-size     16 \
        --workers        4 \
        --seed           42 \
        --cst-root       $CST_ROOT \
        --rgbt-root      $RGBT_ROOT \
        --save-epoch-ckpts \
        --save-epoch-interval 1

echo "=============================="
echo "Done: $(date)"
echo "=============================="
