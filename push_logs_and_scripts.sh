#!/usr/bin/env bash
# Run from Git Bash inside UvA-Masters-Thesis/:
#   cd "/c/Users/zangt/OneDrive/Desktop/mystuff/UvA_Masters/Thesis/UvA-Masters-Thesis"
#   bash push_logs_and_scripts.sh

set -e

rm -f .git/index.lock

# ── READMEs ──────────────────────────────────────────────────────────────────
git add README.md
git add logs/README.md

# ── Canonical result logs ─────────────────────────────────────────────────────
git add \
  "logs/stage1_rerun_23297129.out" \
  "logs/stage2_seed42_23349658.err" \
  "logs/stage2_seed123_23349659.err" \
  "logs/stage2_seed999_23349660.err" \
  "logs/stage3_naive_23394054.err" \
  "logs/s2_analysis_h100_23381446.out" \
  "logs/build_buffers_23391452.out" \
  "logs/drift_23392372.out" \
  "logs/eval_t1_23821658.out" \
  "logs/track_cst_23824125.out" \
  "logs/stage2_22688135.out" \
  "logs/stage3_herding_23425651.err" \
  "logs/stage3_herding_23443056.err" \
  "logs/stage3_herding_23449442.err"

# ── Updated training scripts ──────────────────────────────────────────────────
git add \
  src/train_stage1.py \
  src/train_stage2_ddp.py \
  src/train_stage3.py

# ── Evaluation & analysis ─────────────────────────────────────────────────────
git add \
  src/eval_stage1.py \
  src/eval_full_analysis.py \
  src/eval_tracking_cst.py \
  src/build_herding_buffer.py \
  src/parameter_drift.py \
  src/compute_drift_s2s3.py \
  src/plot_multirun_ci.py \
  src/plot_scale_distribution.py \
  src/plot_training_analysis.py \
  src/visualise_detections_rgbt.py \
  src/json2yolo.py

# ── SLURM scripts ─────────────────────────────────────────────────────────────
git add \
  src/run_stage1_ddp.sh \
  src/run_stage1_rerun.sh \
  src/run_stage2_ddp.sh \
  src/run_multirun_stage2.sh \
  src/run_stage3_naive.sh \
  src/run_stage3_herding.sh \
  src/run_eval.sh \
  src/run_eval_stage1.sh \
  src/run_tracking_eval.sh \
  src/run_vis.sh \
  src/run_vis_stage1.sh \
  src/run_vis_stage2.sh \
  src/run_analysis_drift.sh \
  src/run_analysis_stage1.sh \
  src/run_analysis_stage2.sh \
  src/run_analysis_stage2_h100.sh \
  src/run_build_buffers.sh

# ── Dataset loaders ───────────────────────────────────────────────────────────
git add \
  src/datasets/__init__.py \
  src/datasets/antiuav_rgbt.py \
  src/datasets/antiuav410.py \
  src/datasets/cst.py \
  src/datasets/base.py

# ── Commit ────────────────────────────────────────────────────────────────────
git commit -m "Add canonical result logs and sync final working scripts

Logs added (all 3 stages + analysis):
- Stage 1 rerun 23297129: mAP=0.6725 (T1 ceiling)
- Stage 2 seeds 42/123/999: FM=-0.033±0.004
- Stage 2 CI aggregation 23381446: Table 2/3 numbers
- Stage 2 pilot 22688135: single-seed KD validation
- Parameter drift 23392372: cosine sim S1→S2=0.896
- Stage 3 naive 23394054: FM=-0.605, large-target mAP→0.000
- T1 eval after S3 23821658: per-stratum breakdown
- Buffer construction 23391452: 300-exemplar SSH buffer
- CST tracking eval 23824125: SR@0.5=0.012, PR@20=0.219
- Herding attempts 23425651/23443056/23449442: budget exhaustion

Scripts: sync final versions of all training, eval, analysis, and SLURM files"

# ── Push ──────────────────────────────────────────────────────────────────────
git push origin main

echo ""
echo "Done. https://github.com/GiangNguyen06/UvA-Masters-Thesis-UAV"
