#!/usr/bin/env bash
# Push meeting_6 report + new SLURM scripts to GitHub
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo "=== Staging files ==="
git add src/meeting_6.tex
git add src/meeting_2.tex src/meeting_3.tex src/meeting_4.tex src/meeting_5.tex
git add src/run_drift_s2s3.sh src/run_eval_pr_s3.sh
git add README.md

echo "=== Committing ==="
git commit -m "Add meeting 6 report, drift+eval scripts, fix H100 SBU rate in README

- src/meeting_6.tex: Week 24 supervisor feedback analysis
  - Per-layer drift results (job 23937817, S2->S3)
  - AP=0 rules out calibration failure (Point 2)
  - Gradient imbalance argument for normal collapse (Point 3)
  - lambda ablation and distribution shift gaps documented (Points 4, 6)
  - Supporting citations: masip2026face, molahasani2023continual, he2024gradientreweighting
- src/run_drift_s2s3.sh: SLURM script for parameter drift S2->S3 (rome partition)
- src/run_eval_pr_s3.sh: SLURM script for PR curve eval on T1 after Stage 3
- README.md: correct H100 rate (192 SBU/GPU-hour, not 768)"

echo "=== Pushing ==="
git push origin main

echo "=== Done ==="
