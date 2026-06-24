#!/bin/bash
#SBATCH --job-name=drift_s2s3
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/prjs2041/logs/drift_s2s3_%j.out
#SBATCH --error=/projects/prjs2041/logs/drift_s2s3_%j.err

# S2→S3 per-layer-group parameter drift
# CPU only (rome partition) — costs ~1 SBU
# Outputs: backbone vs neck vs head cosine similarity + per-layer dot plots

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/parameter_drift.py

S2=/projects/prjs2041/runs/stage2/seed42/weights/best.pt
S3=/projects/prjs2041/runs/stage3/naive2/weights/best.pt
OUT=/projects/prjs2041/analysis/drift_s2s3

echo "=============================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

if [ ! -f "$S2" ]; then
    echo "ERROR: Stage 2 weights not found at $S2"; exit 1
fi
if [ ! -f "$S3" ]; then
    echo "ERROR: Stage 3 weights not found at $S3"; exit 1
fi

$PYTHON $SCRIPT \
    --stage1   $S2 \
    --stage2   $S3 \
    --label-a  "Stage 2" \
    --label-b  "Stage 3 (naive)" \
    --out-dir  $OUT \
    --device   cpu

echo "=============================="
echo "Done: $(date)"
echo "Results: $OUT/"
echo "Key figures:"
echo "  fig_drift_groups.png  — backbone vs neck vs head drift"
echo "  fig_drift_cos.png     — per-layer cosine similarity"
echo "  drift_stats.csv       — raw per-layer numbers"
echo "=============================="
