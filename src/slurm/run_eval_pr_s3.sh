#!/bin/bash
#SBATCH --job-name=eval_pr_s3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/prjs2041/logs/eval_pr_s3_%j.out
#SBATCH --error=/projects/prjs2041/logs/eval_pr_s3_%j.err

# PR curves + recall + confidence distributions on T1 val after Stage 3
# Addresses supervisor Point 2: calibration failure vs true forgetting
# Single A100, ~128 SBU

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
UAV_CODE=/projects/prjs2041/uav_code

WEIGHTS=/projects/prjs2041/runs/stage3/naive2/weights/best.pt
DATA=/projects/prjs2041/datasets/Anti-UAV-RGBT
OUT=/projects/prjs2041/analysis/stage3_naive/eval_t1_pr

echo "=============================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

$PYTHON $UAV_CODE/eval_full_analysis.py \
    --weights   $WEIGHTS \
    --dataset   rgbt \
    --data-root $DATA \
    --out-dir   $OUT \
    --device    0

echo "=============================="
echo "Done: $(date)"
echo "Results: $OUT/"
echo "Key outputs:"
echo "  fig_pr_curve.png    — full PR curve"
echo "  fig_conf_curve.png  — P/R/F1 vs confidence threshold"
echo "  fig_scale_metrics.png — mAP by stratum"
echo "  scale_metrics.csv   — per-stratum AP/recall numbers"
echo "=============================="
