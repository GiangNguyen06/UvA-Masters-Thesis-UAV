#!/bin/bash
#SBATCH --job-name=grad_diag
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/prjs2041/logs/grad_diag_%j.out
#SBATCH --error=/projects/prjs2041/logs/grad_diag_%j.err

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/grad_starvation_diagnostic.py

CST_ROOT=/projects/prjs2041/datasets/CST-AntiUAV
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

echo "=============================="
echo "Gradient starvation diagnostic"
echo "Job ID  : $SLURM_JOB_ID"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# Probe each Stage 2 seed checkpoint so the result is not single-seed.
for SEED in 42 123 999; do
    WEIGHTS=/projects/prjs2041/runs/stage2/seed${SEED}/weights/best.pt
    if [[ -f "$WEIGHTS" ]]; then
        echo ""
        echo "### Stage 2 seed ${SEED}  ->  $WEIGHTS"
        $PYTHON $SCRIPT \
            --weights     $WEIGHTS \
            --num-batches 60 \
            --batch-size  16 \
            --workers     8 \
            --seed        42 \
            --control-only-large \
            --cst-root    $CST_ROOT \
            --rgbt-root   $RGBT_ROOT \
            --out         /projects/prjs2041/runs/diagnostics/grad_starvation/seed${SEED}
    else
        echo "skip: $WEIGHTS not found"
    fi
done

echo "=============================="
echo "Done: $(date)"
echo "=============================="
