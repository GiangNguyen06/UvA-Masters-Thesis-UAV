# Logs

SLURM output logs from Snellius (SURF HPC). Each `.out` file is the stdout/stderr of one job.

## Logs in this folder

| File | Job ID | Stage | Outcome |
|------|--------|-------|---------|
| `stage1_ddp_22522864.out` | 22522864 | Stage 1 | First complete S1 run — mAP@0.5 = 0.6617 at ep. 54 (A100). Superseded by re-run 23297129. |
| `stage2_22637974.out` | 22637974 | Stage 2 | Early S2 development run. Superseded by multi-seed jobs below. |
| `eval_stage1_22583396.out` | 22583396 | Stage 1 eval | Full evaluation of S1 checkpoint — per-stratum mAP, PR curves. |

## Important logs NOT in this folder (still on Snellius)

These are the canonical result logs referenced in the thesis:

| Job ID | Stage | Notes |
|--------|-------|-------|
| 23297129 | Stage 1 | **Final S1** — patched script, mAP@0.5 = **0.6725** at ep. 49 (A100) |
| 22688135 | Stage 2 | Single-seed pilot, FM = −0.0153 (A100) — superseded |
| 23349180 | Stage 2 | **Seed 42** — FM = −0.037, best ep. 16/32 (H100) |
| 23349181 | Stage 2 | **Seed 123** — FM = −0.028 (H100) |
| 23349182 | Stage 2 | Seed 999 — cancelled at ep. 29, best.pt saved (H100) |
| 23394054 | Stage 3 | **Naive baseline** — FM = −0.605, ep. 3 best (H100) |
| 23340483 | Analysis | Full eval + vis; scale bins corrected post-run |
| 23381446 | Analysis | Multi-seed CI aggregation + per-seed eval (44 min) |
| 23391452 | Analysis | Replay buffer construction (herding + random) |

## Early development logs (kept for traceability, not thesis-relevant)

`stage1_ddp_22487080.out`, `22501402.out`, `22501881.out`, `22505545.out`, `22519342.out` — failed or superseded Stage 1 runs during dataset/DDP debugging.

`gen_masks_npz_22460009.out` — ARD100 motion mask generation (pre-experiment setup).
