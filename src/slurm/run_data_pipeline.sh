#!/bin/bash
# =============================================================
# Data Pipeline — Anti-UAV Master's Thesis
# Runs all data processing steps in the correct order.
#
# Submit with:
#   sbatch run_data_pipeline.sh
#
# Jobs run as a chain: each step only starts after the previous
# one completes successfully.
# =============================================================

# ── Shared settings ───────────────────────────────────────────
PYTHON=/home/knguyen1/.conda/envs/uav_master/bin/python
CODE=/projects/prjs2041/uav_code
LOGS=/projects/prjs2041/logs
mkdir -p $LOGS

# =============================================================
# Job 1a: Extract Anti-UAV-RGBT frames
# =============================================================
JOB1A=$(sbatch --parsable \
  --job-name=extract_rgbt \
  --partition=rome \
  --time=08:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --output=$LOGS/extract_rgbt_%j.out \
  --wrap="
    module load 2023
    module load Miniconda3/23.5.2-0
    source activate uav_master
    echo 'Starting Anti-UAV-RGBT frame extraction...'
    $PYTHON $CODE/extract_antiuav_rgbt.py
    echo 'Anti-UAV-RGBT extraction complete.'
  ")
echo "Submitted Job 1a (extract_rgbt):   job ID $JOB1A"

# =============================================================
# Job 1b: Extract ARD100 frames (runs in parallel with 1a)
# =============================================================
JOB1B=$(sbatch --parsable \
  --job-name=extract_ard100 \
  --partition=rome \
  --time=08:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --output=$LOGS/extract_ard100_%j.out \
  --wrap="
    module load 2023
    module load Miniconda3/23.5.2-0
    source activate uav_master
    echo 'Starting ARD100 frame extraction...'
    $PYTHON $CODE/extract_ard100.py
    echo 'ARD100 extraction complete.'
  ")
echo "Submitted Job 1b (extract_ard100): job ID $JOB1B"

# =============================================================
# Job 2: Prepare ARD100 (waits for Job 1b to finish)
# =============================================================
JOB2=$(sbatch --parsable \
  --job-name=prepare_ard100 \
  --partition=rome \
  --time=02:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --output=$LOGS/prepare_ard100_%j.out \
  --dependency=afterok:$JOB1B \
  --wrap="
    module load 2023
    module load Miniconda3/23.5.2-0
    source activate uav_master
    echo 'Starting ARD100 data preparation...'
    $PYTHON $CODE/prepare_ard100.py
    echo 'ARD100 preparation complete.'
  ")
echo "Submitted Job 2  (prepare_ard100): job ID $JOB2 (waits for $JOB1B)"

# =============================================================
# Job 3: Generate motion masks (waits for Job 2 to finish)
# =============================================================
JOB3=$(sbatch --parsable \
  --job-name=gen_masks \
  --partition=rome \
  --time=12:00:00 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=32G \
  --output=$LOGS/gen_masks_%j.out \
  --dependency=afterok:$JOB2 \
  --wrap="
    module load 2023
    module load Miniconda3/23.5.2-0
    source activate uav_master
    echo 'Starting motion mask generation...'
    $PYTHON $CODE/generate_mask5_fixed.py
    echo 'Mask generation complete.'
  ")
echo "Submitted Job 3  (gen_masks):      job ID $JOB3 (waits for $JOB2)"

# =============================================================
echo ""
echo "Pipeline submitted. Job chain:"
echo "  [$JOB1A] extract_rgbt   ─┐"
echo "  [$JOB1B] extract_ard100 ─┤"
echo "                            ▼"
echo "  [$JOB2]  prepare_ard100 ─┐  (starts after $JOB1B)"
echo "                            ▼"
echo "  [$JOB3]  gen_masks         (starts after $JOB2)"
echo ""
echo "Monitor with:  squeue -u knguyen1"
echo "Check logs in: $LOGS/"
