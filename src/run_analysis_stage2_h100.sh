#!/bin/bash
#SBATCH --job-name=s2_analysis_h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/prjs2041/logs/s2_analysis_h100_%j.out
#SBATCH --error=/projects/prjs2041/logs/s2_analysis_h100_%j.err

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
UAV_CODE=/projects/prjs2041/uav_code
RUNS=/projects/prjs2041/runs/stage2
T1_BASELINE=0.6725

echo "=============================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# ── Copy Anti-UAV410 to local SSD ─────────────────────────────────────────────
echo ""
echo "Copying Anti-UAV410 to \$TMPDIR ..."
T0=$(date +%s)
rsync -a --no-relative /projects/prjs2041/datasets/Anti-UAV410/ $TMPDIR/Anti-UAV410/
echo "Done in $(($(date +%s) - T0))s"

# ── Step 1: CI aggregation (reads results.csv, no GPU needed) ─────────────────
echo ""
echo "=== Step 1: Multi-seed CI plots ==="
$PYTHON $UAV_CODE/plot_multirun_ci.py \
    --runs-root $RUNS \
    --pattern   "seed*" \
    --t1-baseline $T1_BASELINE \
    --out-dir   $RUNS/ci_plots

# ── Step 2: Per-seed training curves ──────────────────────────────────────────
echo ""
echo "=== Step 2: Per-seed training curves ==="
for SEED in seed42 seed123 seed999; do
    CSV=$RUNS/$SEED/results.csv
    if [ -f "$CSV" ]; then
        echo "  Processing $SEED ..."
        $PYTHON $UAV_CODE/plot_training_analysis.py \
            --csv        $CSV \
            --out-dir    $RUNS/$SEED/analysis \
            --t1-baseline $T1_BASELINE
    else
        echo "  [SKIP] $SEED — no results.csv found"
    fi
done

# ── Step 3: Full eval on each seed's best.pt (scale metrics, PR curve, etc.) ──
echo ""
echo "=== Step 3: Full evaluation per seed ==="
for SEED in seed42 seed123 seed999; do
    WEIGHTS=$RUNS/$SEED/weights/best.pt
    if [ -f "$WEIGHTS" ]; then
        echo "  Evaluating $SEED ..."
        $PYTHON $UAV_CODE/eval_full_analysis.py \
            --weights   $WEIGHTS \
            --dataset   uav410 \
            --data-root $TMPDIR/Anti-UAV410 \
            --out-dir   $RUNS/$SEED/analysis \
            --device    0
    else
        echo "  [SKIP] $SEED — no best.pt found"
    fi
done

echo ""
echo "=============================="
echo "Done: $(date)"
echo "CI plots  : $RUNS/ci_plots/"
echo "Per-seed  : $RUNS/seed{42,123,999}/analysis/"
echo "=============================="
