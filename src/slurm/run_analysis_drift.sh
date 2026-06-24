#!/bin/bash
#SBATCH --job-name=drift_s1s2
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/prjs2041/logs/drift_%j.out
#SBATCH --error=/projects/prjs2041/logs/drift_%j.err

set -euo pipefail

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/parameter_drift.py
S1=/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt

echo "=============================="
echo "Parameter Drift Analysis: Stage 1 → Stage 2"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "=============================="

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

if [ ! -f "$S1" ]; then
    echo "ERROR: Stage 1 weights not found at $S1"
    exit 1
fi

for SEED in seed42 seed123 seed999; do
    S2=/projects/prjs2041/runs/stage2/$SEED/weights/best.pt
    if [ ! -f "$S2" ]; then
        echo "[SKIP] $SEED — no best.pt found at $S2"
        continue
    fi

    echo ""
    echo "--- $SEED ---"
    $PYTHON $SCRIPT \
        --stage1  $S1 \
        --stage2  $S2 \
        --out-dir /projects/prjs2041/runs/stage2/$SEED/drift \
        --device  cpu
done

echo ""
echo "=============================="
echo "Done: $(date)"
echo "Results per seed:"
echo "  /projects/prjs2041/runs/stage2/seed42/drift/"
echo "  /projects/prjs2041/runs/stage2/seed123/drift/"
echo "  /projects/prjs2041/runs/stage2/seed999/drift/"
echo ""
echo "Key figure: fig_drift_groups.png"
echo "  backbone = low drift → Stage 1 features preserved"
echo "  head     = higher drift → adapted to Anti-UAV410"
echo "=============================="
