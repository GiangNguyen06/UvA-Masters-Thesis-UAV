#!/bin/bash
# =============================================================
# run_stage1.sh
# Stage 1: Train YOLOMG on Anti-UAV-RGBT IR stream
#
# Submit with:
#   sbatch run_stage1.sh
#
# Outputs:
#   /projects/prjs2041/runs/stage1/antiuav_rgbt*/
#       weights/best.pt       ← use as --weights for Stage 2
#       weights/last.pt
#       results.csv
#       stage1_mAP.txt        ← T1 mAP for Forgetting Measure
#   /projects/prjs2041/logs/stage1_<JOBID>.out
# =============================================================
#SBATCH --job-name=stage1_yolomg
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/projects/prjs2041/logs/stage1_%j.out

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/train_stage1.py
LOGS=/projects/prjs2041/logs
SRC_DATA=/projects/prjs2041/datasets/Anti-UAV-RGBT

# Local NVMe scratch: SLURM creates /scratch-local/$USER.$SLURM_JOB_ID automatically.
# This is fast node-local NVMe, unlike $TMPDIR which is GPFS (shared network FS).
LOCAL_SCRATCH=/scratch-local/${USER}.${SLURM_JOB_ID}
LOCAL_DATA=${LOCAL_SCRATCH}/Anti-UAV-RGBT

mkdir -p $LOGS

echo "========================================"
echo "Stage 1: YOLOMG on Anti-UAV-RGBT"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $CUDA_VISIBLE_DEVICES"
echo "Time:    $(date)"
echo "========================================"
echo ""

# ── Copy dataset to local NVMe scratch ────────────────────────────────────
echo "Local scratch space:"
df -h $LOCAL_SCRATCH

echo "Copying Anti-UAV-RGBT to local NVMe ($LOCAL_DATA)..."
cp -r $SRC_DATA $LOCAL_DATA
echo "Copy complete at $(date). Size on disk:"
du -sh $LOCAL_DATA
echo ""

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

$PYTHON $SCRIPT \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --workers 4 \
    --device 0 \
    --dataset-root $LOCAL_DATA \
    --name antiuav_rgbt

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "Check results: /projects/prjs2041/runs/stage1/"
echo "========================================"
