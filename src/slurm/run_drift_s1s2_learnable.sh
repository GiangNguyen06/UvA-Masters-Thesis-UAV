#!/bin/bash
#SBATCH --job-name=drift_s1s2_learn
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/prjs2041/logs/drift_s1s2_learnable_%j.out
#SBATCH --error=/projects/prjs2041/logs/drift_s1s2_learnable_%j.err

# S1→S2 per-layer-group parameter drift — LEARNABLE PARAMETERS ONLY
# Regenerates the S1→S2 cosine number (thesis Table 6: currently 0.911)
# using only requires_grad=True tensors, so both table entries (S1→S2 and
# S2→S3) are on the same footing (learnable-only, unweighted tensor mean).
#
# CPU only (rome partition) — ~1 SBU
#
# Outputs go to /projects/prjs2041/analysis/drift_s1s2/ with _learnable suffix:
#   drift_stats_learnable.csv
#   fig_drift_cos_learnable.png
#   fig_drift_groups_learnable.png
#   etc.

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/parameter_drift.py

S1=/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt
S2=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
OUT=/projects/prjs2041/analysis/drift_s1s2

echo "=============================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "Mode    : learnable-only (requires_grad=True)"
echo "Transition: S1 -> S2"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

if [ ! -f "$S1" ]; then
    echo "ERROR: Stage 1 weights not found at $S1"; exit 1
fi
if [ ! -f "$S2" ]; then
    echo "ERROR: Stage 2 weights not found at $S2"; exit 1
fi

mkdir -p $OUT

$PYTHON $SCRIPT \
    --stage1        $S1 \
    --stage2        $S2 \
    --label-a       "Stage 1" \
    --label-b       "Stage 2" \
    --out-dir       $OUT \
    --device        cpu \
    --learnable-only

echo "=============================="
echo "Done: $(date)"
echo "Results: $OUT/"
echo "Key outputs:"
echo "  drift_stats_learnable.csv       — raw per-layer numbers"
echo "  fig_drift_groups_learnable.png  — backbone vs neck vs head"
echo "Per-group unweighted cosine printed to stdout above."
echo "=============================="
