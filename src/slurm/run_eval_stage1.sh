#!/bin/bash
#SBATCH --job-name=eval_stage1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/prjs2041/logs/eval_stage1_%j.out

echo "========================================"
echo "Stage 1 Eval: YOLOMG on Anti-UAV-RGBT"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Time:    $(date)"
echo "========================================"

# ── Copy dataset to GPFS scratch ─────────────────────────────────────────────
SCRATCH=/gpfs/scratch1/shared/${USER}
DATASET_SCRATCH=${SCRATCH}/Anti-UAV-RGBT

echo "Copying Anti-UAV-RGBT to GPFS scratch..."
mkdir -p ${SCRATCH}
rsync -a --info=progress2 \
    /projects/prjs2041/datasets/Anti-UAV-RGBT/ \
    ${DATASET_SCRATCH}/
echo "Done."

# ── Run eval ──────────────────────────────────────────────────────────────────
PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python

${PYTHON} /projects/prjs2041/uav_code/eval_stage1.py \
    --weights  /projects/prjs2041/runs/stage1/antiuav_rgbt14/weights/best.pt \
    --dataset-root ${DATASET_SCRATCH} \
    --batch-size 32 \
    --workers 4 \
    --device 0

echo "Done: $(date)"
