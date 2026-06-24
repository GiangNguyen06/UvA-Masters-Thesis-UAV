#!/usr/bin/env bash
# Run from Git Bash inside UvA-Masters-Thesis/:
#   cd "/c/Users/zangt/OneDrive/Desktop/mystuff/UvA_Masters/Thesis/UvA-Masters-Thesis"
#   bash cleanup_and_push.sh

set -e

rm -f .git/index.lock

# ── REMOVE: tex/notes (not code) ────────────────────────────────────────────
git rm --cached -f \
  "src/meeting_2.tex" \
  "src/methodology_draft.tex" \
  "src/rq_answers.tex" \
  "src/week14.tex" \
  "src/sections/methodology.tex"

# ── REMOVE: unused Python scripts ───────────────────────────────────────────
# Motion mask generation — motion channel was zeroed throughout all stages
git rm --cached -f \
  "src/generate_mask5_fixed.py" \
  "src/generate_masks_npz.py"

# Data extraction — setup-only, not part of training/eval pipeline
git rm --cached -f \
  "src/extract_antiuav_rgbt.py" \
  "src/extract_ard100.py" \
  "src/extract_frames_antiuav_rgbt.py" \
  "src/extract_frames.sh"

# ARD100 prep — ARD100 not used in any experiment
git rm --cached -f \
  "src/prepare_ard100.py" \
  "src/datasets/ard100.py"

# Alternative/superseded loaders and scripts
git rm --cached -f \
  "src/datasets/antiuav_rgbt_frames.py" \
  "src/train_stage2.py" \
  "src/audit_datasets.py" \
  "src/test_datasets.py"

# Superseded single-GPU SLURM scripts
git rm --cached -f \
  "src/run_stage1.sh" \
  "src/run_stage2.sh" \
  "src/run_data_pipeline.sh" \
  "src/run_generate_masks.sh"

# ── REMOVE: duplicate / early log files ─────────────────────────────────────
git rm --cached -f \
  "logs/eval_stage1_22582606.out" \
  "logs/eval_stage1_22582761.out" \
  "logs/gen_masks_npz_22460009 (1).out" \
  "logs/stage1_ddp_22487080 (1).out" \
  "logs/stage1_ddp_22522864 (1).out" \
  "logs/stage1_ddp_22522864 (2).out" \
  "logs/stage1_ddp_22522864 (3).out" \
  "logs/stage1_ddp_22522864 (4).out" \
  "logs/stage1_ddp_22522864 (5).out" \
  "logs/stage1_ddp_22522864-2.out"

# ── ADD: README files ────────────────────────────────────────────────────────
git add README.md
git add logs/README.md

# ── ADD: canonical result logs ───────────────────────────────────────────────
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

# ── ADD: core training scripts ───────────────────────────────────────────────
git add \
  src/train_stage1.py \
  src/train_stage2_ddp.py \
  src/train_stage3.py

# ── ADD: evaluation & analysis ───────────────────────────────────────────────
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

# ── ADD: SLURM scripts actually used ─────────────────────────────────────────
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

# ── ADD: dataset loaders actually used ───────────────────────────────────────
git add \
  src/datasets/__init__.py \
  src/datasets/antiuav_rgbt.py \
  src/datasets/antiuav410.py \
  src/datasets/cst.py \
  src/datasets/base.py

# ── Commit ───────────────────────────────────────────────────────────────────
git commit -m "Sync final working scripts; remove unused files

Keep only what was actually used in the three-stage continual learning experiments:
- Training: train_stage1, train_stage2_ddp, train_stage3
- Eval: eval_full_analysis, eval_stage1, eval_tracking_cst
- Analysis: parameter_drift, compute_drift_s2s3, plot_*
- Buffer: build_herding_buffer
- Datasets: antiuav_rgbt, antiuav410, cst, base
- SLURM: stage1-rerun, stage2-ddp, multirun, stage3, eval, vis, analysis, buffers

Remove: motion mask scripts, ARD100 prep, data extraction, superseded loaders,
single-GPU scripts, duplicate logs, tex meeting notes"

# ── Push ─────────────────────────────────────────────────────────────────────
git push origin main

echo ""
echo "Done. https://github.com/GiangNguyen06/UvA-Masters-Thesis-UAV"
