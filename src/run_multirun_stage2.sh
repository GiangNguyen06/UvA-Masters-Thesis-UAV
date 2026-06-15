#!/bin/bash
# ============================================================================
# run_multirun_stage2.sh
# ============================================================================
# Submit THREE independent Stage 2 runs with different random seeds
# to produce confidence intervals for thesis reporting.
#
# Usage (on Snellius login node):
#   bash run_multirun_stage2.sh
#
# Each job:
#   - Copies Anti-UAV410 + Anti-UAV-RGBT data to $TMPDIR (local NVMe SSD)
#   - Runs 4-GPU DDP Stage 2 with a different seed
#   - Saves per-epoch checkpoints every 5 epochs
#   - Saves results.csv with P/R/F1/mAP per epoch for both T1 and T2
#
# Results land in:
#   /projects/prjs2041/runs/stage2/seed42/
#   /projects/prjs2041/runs/stage2/seed123/
#   /projects/prjs2041/runs/stage2/seed999/
#
# After all three finish:
#   python /projects/prjs2041/uav_code/plot_multirun_ci.py \
#       --runs-root /projects/prjs2041/runs/stage2 \
#       --pattern   "seed*" \
#       --t1-baseline 0.6725
# ============================================================================

# ── Updated to use antiuav_rgbt15 (mAP@0.5 = 0.6725, epoch 49) ──────────────
STAGE1_WEIGHTS="/projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt"
TORCHRUN="/home/knguyen1/.conda/envs/uav_master/bin/torchrun"
SCRIPT="/projects/prjs2041/uav_code/train_stage2_ddp.py"
SAVE_ROOT="/projects/prjs2041/runs/stage2"
LOGS_DIR="/projects/prjs2041/logs"

SEEDS=(42 123 999)
NAMES=("seed42" "seed123" "seed999")

mkdir -p "$LOGS_DIR"

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    NAME=${NAMES[$i]}

    echo "Submitting Stage 2  seed=${SEED}  name=${NAME} ..."

    sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=s2_${NAME}
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=${LOGS_DIR}/stage2_${NAME}_%j.out
#SBATCH --error=${LOGS_DIR}/stage2_${NAME}_%j.err

set -euo pipefail

echo "=============================="
echo "Job ID      : \$SLURM_JOB_ID"
echo "Node        : \$SLURMD_NODENAME"
echo "GPUs        : \$CUDA_VISIBLE_DEVICES"
echo "Seed        : ${SEED}"
echo "Name        : ${NAME}"
echo "Teacher     : ${STAGE1_WEIGHTS}"
echo "Started     : \$(date)"
echo "=============================="

# ── Load conda environment ────────────────────────────────────────────────────
module load 2023
module load Miniconda3/23.5.2-0
source activate uav_master

# ── Copy datasets to local NVMe SSD ──────────────────────────────────────────
echo ""
echo "Copying Anti-UAV410 to \$TMPDIR ..."
T0=\$(date +%s)
rsync -a --no-relative /projects/prjs2041/datasets/Anti-UAV410/ \$TMPDIR/Anti-UAV410/
echo "Done in \$(((\$(date +%s) - T0)))s  |  \$(du -sh \$TMPDIR/Anti-UAV410 | cut -f1)"

echo "Copying Anti-UAV-RGBT to \$TMPDIR ..."
T0=\$(date +%s)
rsync -a --no-relative /projects/prjs2041/datasets/Anti-UAV-RGBT/ \$TMPDIR/Anti-UAV-RGBT/
echo "Done in \$(((\$(date +%s) - T0)))s  |  \$(du -sh \$TMPDIR/Anti-UAV-RGBT | cut -f1)"

echo "Data ready: \$(date)"

# ── Run Stage 2 DDP ───────────────────────────────────────────────────────────
echo ""
echo "Starting Stage 2 DDP training ..."

${TORCHRUN} \
    --standalone \
    --nproc_per_node=4 \
    ${SCRIPT} \
        --weights             ${STAGE1_WEIGHTS} \
        --name                ${NAME} \
        --seed                ${SEED} \
        --save-epoch-ckpts \
        --save-epoch-interval 5 \
        --kd-weight           1.0 \
        --epochs              50 \
        --batch-size          16 \
        --workers             4

echo ""
echo "=============================="
echo "Stage 2 complete: \$(date)"
echo "Results in: ${SAVE_ROOT}/${NAME}/"
echo "=============================="
SLURM

    echo "  → Submitted  seed=${SEED}"
    echo ""
done

echo "All 3 seeds submitted."
echo "Monitor with:  squeue -u \$USER"
echo ""
echo "After all finish, run:"
echo "  python /projects/prjs2041/uav_code/plot_multirun_ci.py \\"
echo "      --runs-root ${SAVE_ROOT} --pattern 'seed*' --t1-baseline 0.6725"
