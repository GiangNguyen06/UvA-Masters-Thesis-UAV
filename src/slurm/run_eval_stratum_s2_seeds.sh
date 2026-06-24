#!/bin/bash
#SBATCH --job-name=stratum_s2_seeds
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/prjs2041/logs/stratum_s2_seeds_%j.out
#SBATCH --error=/projects/prjs2041/logs/stratum_s2_seeds_%j.err

# Per-stratum T1 mAP on Anti-UAV-RGBT val for Stage 2 seeds 123 and 999.
# Completes the "After S2" column in the per-stratum table so it reports
# mean ± std across all three seeds (matching how Stage 2 FM is reported).

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/eval_stratum_t1.py
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

S1=/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt
S2_S123=/projects/prjs2041/runs/stage2/seed123/weights/best.pt
S2_S999=/projects/prjs2041/runs/stage2/seed999/weights/best.pt

echo "=============================="
echo "Per-stratum T1 eval — Stage 2 seeds 123 and 999"
echo "Job ID  : $SLURM_JOB_ID"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

echo "--- Seed 123 ---"
$PYTHON $SCRIPT \
    --s1 "$S1" \
    --s2 "$S2_S123" \
    --split val \
    --batch-size 32 \
    --workers 8 \
    --rgbt-root "$RGBT_ROOT" \
    --out /projects/prjs2041/runs/diagnostics/stratum_t1/seed123

echo "--- Seed 999 ---"
$PYTHON $SCRIPT \
    --s1 "$S1" \
    --s2 "$S2_S999" \
    --split val \
    --batch-size 32 \
    --workers 8 \
    --rgbt-root "$RGBT_ROOT" \
    --out /projects/prjs2041/runs/diagnostics/stratum_t1/seed999

echo "=============================="
echo "Done: $(date)"
echo "=============================="
