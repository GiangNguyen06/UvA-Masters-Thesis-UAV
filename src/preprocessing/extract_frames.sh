#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rome
#SBATCH --output=/projects/prjs2041/logs/extract_frames_%j.out

module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
CODE_DIR=/projects/prjs2041/uav_code

mkdir -p /projects/prjs2041/logs

echo "========================================"
echo "Starting Anti-UAV-RGBT frame extraction"
echo "========================================"
$PYTHON $CODE_DIR/extract_antiuav_rgbt.py

echo ""
echo "========================================"
echo "Starting ARD100 frame extraction"
echo "========================================"
$PYTHON $CODE_DIR/extract_ard100.py

echo ""
echo "All extractions complete."
