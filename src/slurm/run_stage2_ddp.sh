#!/bin/bash
# =============================================================
# run_stage2.sh
# Stage 2: Teacher-Student UDA — fine-tune YOLOMG on Anti-UAV410
# 4×A100 DDP via torchrun --standalone --nproc_per_node=4
#
# Submit with:
#   sbatch run_stage2.sh
#
# Outputs:
#   /projects/prjs2041/runs/stage2/antiuav410*/
#       weights/best.pt         ← use as --weights for Stage 3
#       weights/last.pt
#       results.csv
#       stage2_t2_mAP.txt       ← best mAP on AntiUAV410 val
#       stage2_t1_mAP.txt       ← T1 mAP after T2 (FM numerator)
#   /projects/prjs2041/logs/stage2_<JOBID>.out
#
# Forgetting Measure:
#   FM = stage2_t1_mAP.txt − stage1_mAP.txt   (negative → forgetting)
# =============================================================
#SBATCH --job-name=stage2_ddp
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:4
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/projects/prjs2041/logs/stage2_%j.out

STAGE1_WEIGHTS=/projects/prjs2041/runs/stage1/antiuav_rgbt14/weights/best.pt

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
TORCHRUN=/home/knguyen1/.conda/envs/uav_master/bin/torchrun
SCRIPT=/projects/prjs2041/uav_code/train_stage2_ddp.py
LOGS=/projects/prjs2041/logs

mkdir -p $LOGS

echo "========================================"
echo "Stage 2 DDP: Teacher-Student UDA on Anti-UAV410"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo "Weights: $STAGE1_WEIGHTS"
echo "Time:    $(date)"
echo "========================================"
echo ""

$TORCHRUN --standalone --nproc_per_node=4 $SCRIPT \
    --weights    $STAGE1_WEIGHTS \
    --epochs     50 \
    --batch-size 16 \
    --imgsz      640 \
    --workers    4 \
    --kd-weight  1.0 \
    --name       antiuav410

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "Check results: /projects/prjs2041/runs/stage2/"
echo "========================================"
