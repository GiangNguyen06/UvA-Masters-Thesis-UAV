#!/bin/bash
#SBATCH --job-name=eval_t1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/prjs2041/logs/eval_t1_%j.out
#SBATCH --error=/projects/prjs2041/logs/eval_t1_%j.err

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

$PYTHON $UAV_CODE/eval_full_analysis.py \
    --weights   /projects/prjs2041/runs/stage3/naive2/weights/best.pt \
    --dataset   rgbt \
    --data-root /projects/prjs2041/datasets/Anti-UAV-RGBT \
    --out-dir   /projects/prjs2041/analysis/stage3_naive/eval_t1 \
    --device    0

echo "=============================="
echo "Done: $(date)"
echo "=============================="
