#!/bin/bash
#SBATCH --job-name=s1_analysis
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/prjs2041/logs/s1_analysis_%j.out
#SBATCH --error=/projects/prjs2041/logs/s1_analysis_%j.err

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
WEIGHTS=/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt
UAV_CODE=/projects/prjs2041/uav_code
OUT_DIR=/projects/prjs2041/runs/stage1/antiuav_rgbt15/analysis

echo "=============================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# ── Copy Anti-UAV-RGBT to local SSD ──────────────────────────────────────────
echo ""
echo "Copying Anti-UAV-RGBT to \$TMPDIR ..."
T0=$(date +%s)
rsync -a --no-relative /projects/prjs2041/datasets/Anti-UAV-RGBT/ $TMPDIR/Anti-UAV-RGBT/
echo "Done in $(($(date +%s) - T0))s"

# ── 1. Training curve figures (no GPU needed — reads results.csv only) ────────
echo ""
echo "=== Step 1: Training curve analysis ==="
$PYTHON $UAV_CODE/plot_training_analysis.py \
    --csv     /projects/prjs2041/runs/stage1/antiuav_rgbt15/results.csv \
    --out-dir $OUT_DIR

# ── 2. Full evaluation: PR curve, scale-stratified mAP, per-sequence ─────────
# Uses fixed UAV size bins: tiny<16px, small 16-32px, normal 32-64px, large>64px
echo ""
echo "=== Step 2: Full evaluation (fixed scale bins) ==="
$PYTHON $UAV_CODE/eval_full_analysis.py \
    --weights   $WEIGHTS \
    --dataset   rgbt \
    --data-root $TMPDIR/Anti-UAV-RGBT \
    --out-dir   $OUT_DIR \
    --device    0

# ── 3. Bounding-box visualisation ─────────────────────────────────────────────
echo ""
echo "=== Step 3: Bounding-box visualisation ==="
$PYTHON $UAV_CODE/visualise_detections_rgbt.py \
    --weights   $WEIGHTS \
    --data-root $TMPDIR/Anti-UAV-RGBT \
    --out-dir   $OUT_DIR/vis \
    --n-seqs    8 \
    --n-frames  80 \
    --device    0

echo ""
echo "=============================="
echo "Done: $(date)"
echo "Results in: $OUT_DIR"
echo "=============================="
