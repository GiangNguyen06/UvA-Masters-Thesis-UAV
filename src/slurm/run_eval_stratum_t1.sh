#!/bin/bash
#SBATCH --job-name=stratum_t1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/prjs2041/logs/stratum_t1_%j.out
#SBATCH --error=/projects/prjs2041/logs/stratum_t1_%j.err

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/eval_stratum_t1.py
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

# Adjust checkpoint paths below if they differ from the defaults.
S1=/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt
S2=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
S3=/projects/prjs2041/runs/stage3/naive2/weights/best.pt

echo "=============================="
echo "Per-stratum T1 evaluation (S1 / S2 / S3)"
echo "Job ID  : $SLURM_JOB_ID"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

$PYTHON $SCRIPT \
    --s1 "$S1" \
    --s2 "$S2" \
    --s3 "$S3" \
    --split val \
    --batch-size 32 \
    --workers 8 \
    --rgbt-root "$RGBT_ROOT" \
    --out /projects/prjs2041/runs/diagnostics/stratum_t1

echo "=============================="
echo "Done: $(date)"
echo "=============================