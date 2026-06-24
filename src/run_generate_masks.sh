#!/bin/bash
# =============================================================
# run_generate_masks.sh
# Precompute FD5 motion masks for all ARD100 sequences.
#
# Submit with:
#   sbatch run_generate_masks.sh
# =============================================================
#SBATCH --job-name=gen_masks_npz
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/projects/prjs2041/logs/gen_masks_npz_%j.out

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
SCRIPT=/projects/prjs2041/uav_code/generate_masks_npz.py
LOGS=/projects/prjs2041/logs

mkdir -p $LOGS

echo "Starting FD5 mask generation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Time:   $(date)"
echo ""

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

$PYTHON $SCRIPT

echo ""
echo "Finished at: $(date)"
