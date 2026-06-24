#!/bin/bash
#SBATCH --job-name=build_buf_train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/prjs2041/logs/build_buf_train_%j.out
#SBATCH --error=/projects/prjs2041/logs/build_buf_train_%j.err

# Rebuilds herding and random-stratified buffers from the TRAINING split
# of Anti-UAV-RGBT, fixing the train/val overlap present in the original
# buffer (which was built from val, the same split used for T1 evaluation).

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/build_herding_buffer.py
WEIGHTS=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
OUT_DIR=/projects/prjs2041/runs/stage2/seed42
RGBT_ROOT=/projects/prjs2041/datasets/Anti-UAV-RGBT

echo "=============================="
echo "Build replay buffers from TRAIN split"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

echo "--- Herding buffer (train split) ---"
$PYTHON $SCRIPT \
    --weights       $WEIGHTS \
    --data-root     $RGBT_ROOT \
    --split         train \
    --mode          herding \
    --k-per-stratum 75 \
    --out           $OUT_DIR/herding_buffer_train.pt \
    --device        0

echo "--- Random-stratified buffer (train split) ---"
$PYTHON $SCRIPT \
    --weights       $WEIGHTS \
    --data-root     $RGBT_ROOT \
    --split         train \
    --mode          random \
    --k-per-stratum 75 \
    --out           $OUT_DIR/herding_buffer_random_train.pt \
    --device        0

echo "=============================="
echo "Output files:"
ls -lh $OUT_DIR/herding_buffer_*train*.pt
echo "Done: $(date)"
echo "=============================="
