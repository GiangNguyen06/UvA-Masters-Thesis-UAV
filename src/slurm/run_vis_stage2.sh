#!/bin/bash
#SBATCH --job-name=vis_s2
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/prjs2041/logs/vis_s2_%j.out
#SBATCH --error=/projects/prjs2041/logs/vis_s2_%j.err

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

# Stage 2 weights — after KD fine-tuning (T1 FM = -0.033, 95% retention)
$PYTHON $UAV_CODE/visualise_detections_rgbt.py \
    --weights   /projects/prjs2041/runs/stage2/seed42/weights/best.pt \
    --data-root /projects/prjs2041/datasets/Anti-UAV-RGBT \
    --out-dir   /projects/prjs2041/analysis/vis_comparison/stage2 \
    --n-seqs    5 \
    --n-frames  60 \
    --device    0 \
    --no-video

echo "=============================="
echo "Done: $(date)"
echo "Frames: /projects/prjs2041/analysis/vis_comparison/stage2/frames/"
echo "=============================="
