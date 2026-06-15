#!/bin/bash
#SBATCH --job-name=build_buffers
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/prjs2041/logs/build_buffers_%j.out
#SBATCH --error=/projects/prjs2041/logs/build_buffers_%j.err

set -euo pipefail

SCRIPT=/projects/prjs2041/uav_code/build_herding_buffer.py
WEIGHTS=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
OUT_DIR=/projects/prjs2041/runs/stage2/seed42

echo "=============================="
echo "Build Replay Buffers (Stage 2 features)"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

# ── Sanity check ──────────────────────────────────────────────────────────────
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: Stage 2 weights not found at $WEIGHTS"
    exit 1
fi
echo "Stage 2 weights: $WEIGHTS"

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# ── Copy dataset to local SSD ─────────────────────────────────────────────────
echo ""
echo "Copying Anti-UAV-RGBT to \$TMPDIR ..."
rsync -a --no-relative /projects/prjs2041/datasets/Anti-UAV-RGBT/ $TMPDIR/Anti-UAV-RGBT/
echo "Data ready: $(date)"

# ── Build herding buffer (thesis contribution) ────────────────────────────────
echo ""
echo "--- Building HERDING buffer ---"
python $SCRIPT \
    --weights       $WEIGHTS \
    --data-root     $TMPDIR/Anti-UAV-RGBT \
    --mode          herding \
    --k-per-stratum 75 \
    --out           $OUT_DIR/herding_buffer.pt \
    --device        0

echo "Herding buffer done: $(date)"

# ── Build random stratified buffer (ablation) ─────────────────────────────────
echo ""
echo "--- Building RANDOM STRATIFIED buffer (ablation) ---"
python $SCRIPT \
    --weights       $WEIGHTS \
    --data-root     $TMPDIR/Anti-UAV-RGBT \
    --mode          random \
    --k-per-stratum 75 \
    --out           $OUT_DIR/herding_buffer_random.pt \
    --device        0

echo "Random buffer done: $(date)"

# ── Verify outputs ────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "Output files:"
ls -lh $OUT_DIR/herding_buffer*.pt
echo ""
echo "Herding buffer stats:"
cat $OUT_DIR/herding_buffer.txt
echo ""
echo "Random buffer stats:"
cat $OUT_DIR/herding_buffer_random.txt
echo ""
echo "All done: $(date)"
echo "=============================="
