#!/bin/bash
#SBATCH --job-name=track_cst
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/prjs2041/logs/track_cst_%j.out
#SBATCH --error=/projects/prjs2041/logs/track_cst_%j.err

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
UAV_CODE=/projects/prjs2041/uav_code

echo "=============================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# Stage 3 naive checkpoint (post-T3, shows catastrophic forgetting of CST large targets)
$PYTHON $UAV_CODE/eval_tracking_cst.py \
    --weights  /projects/prjs2041/runs/stage3/naive2/weights/best.pt \
    --cst-root /projects/prjs2041/datasets/CST-AntiUAV \
    --split    val \
    --out-dir  /projects/prjs2041/analysis/tracking_eval \
    --device   0

echo "=============================="
echo "Done: $(date)"
echo "Outputs: /projects/prjs2041/analysis/tracking_eval/"
echo "=============================="
