# Logs

SLURM output logs from Snellius (SURF HPC). Files named `.err` contain the actual epoch-by-epoch training output — the training scripts write via Python `logging` which routes to stderr. Files named `.out` contain job setup and final summaries.

---

## Stage 1 — Supervised Training on Anti-UAV-RGBT

### `stage1_rerun_23297129.out` ✅ CANONICAL
**Job:** 23297129 | A100 partition, 4×A100  
**Result:** mAP@0.5 = **0.6725** at epoch 49/100 (early stopping patience=30)

This is the T1 ceiling used for all Forgetting Measure computations: `FM = mAP_T1_after_Tk − 0.6725`

```
Best mAP@0.5 = 0.6725
Weights: /projects/prjs2041/runs/stage1/antiuav_rgbt15/weights/best.pt
Early stopping triggered at epoch 79 (best at epoch 49)
```

---

## Stage 2 — Knowledge Distillation on Anti-UAV410 (3 seeds)

Loss: `L_total = L_det + 1.0 × L_kd` where `L_kd` = MSE on P3/P4/P5 grids vs frozen Stage 1 teacher.

### `stage2_seed42_23349658.err` ✅ PRIMARY SEED
**Job:** 23349658 | H100 partition, 4×H100  
**Result:** T2 mAP = 0.4264, T1 retained = 0.6658, **FM = −0.037**, best epoch 16/32

```
Best T2 mAP@0.5 (AntiUAV410)  = 0.4264
Best T1 mAP@0.5 (AntiUAVRGBT) = 0.6658
Best epoch: 16  (early stopping patience=15)
```

### `stage2_seed123_23349659.err` ✅ SEED 123
**Job:** 23349659 | H100 partition, 4×H100  
**Result:** T2 mAP = 0.4304, T1 retained = 0.6647, **FM = −0.028**, best epoch 15/32

### `stage2_seed999_23349660.err` ⚠️ CANCELLED ep 29
**Job:** 23349660 | H100 partition, 4×H100  
**Status:** Cancelled at epoch 29 by wall-time. Best checkpoint at ep. 29 was saved and used in CI aggregation below.

### `s2_analysis_h100_23381446.out` ✅ CI AGGREGATION
**Job:** 23381446 | `run_analysis_stage2_h100.sh` → `plot_multirun_ci.py`  
**Result:** All 3 seeds aggregated → **FM = −0.033 ± 0.004** (95% CI), T1 retained = **0.640**

Also contains scale-stratified mAP for seed-42 checkpoint on T1 val:
```
tiny=0.0012  small=0.2773  normal=0.6821  large=0.2828
```

### `stage2_22688135.out` 📋 SINGLE-SEED PILOT (superseded)
**Job:** 22688135 | A100 partition, single GPU  
**Result:** FM = −0.0153 at best epoch 10/26 — validated KD concept before DDP multi-seed run.

---

## Parameter Drift Analysis

### `drift_23392372.out` ✅ COSINE SIMILARITY S1→S2
**Job:** 23392372 | `run_analysis_drift.sh` → `parameter_drift.py`  
**Result:** Mean cosine similarity S1→S2 = **0.896** across 335 parameter tensors

```
Global cosine similarity: mean = 0.895565
min = -0.069  (model.4.spatial_attention.conv.weight)
```

Backbone shows low drift (features preserved), head shows higher drift (adapted to Anti-UAV410). Contrast with S2→S3 cosine sim = **0.967** in `compute_drift_s2s3.py` output — weights barely moved but T1 mAP collapsed, confirming gradient starvation rather than bulk overwriting.

---

## Stage 3 — Naive Baseline on CST Anti-UAV

### `stage3_naive_23394054.err` ✅ CANONICAL
**Job:** 23394054 | H100 partition, 4×H100  
**Result:** FM = **−0.605** at best epoch 3, T3 mAP (CST) = 0.0829

```
fm_abs at best T3 epoch = -0.6045  (vs T1 ceiling 0.6725)
T1 mAP at epoch 18 = 0.0168
T1 by stratum at ep 18: tiny=0.0002  small=0.0192  normal=0.0190  large=0.0000
```

### `eval_t1_23821658.out` ✅ T1 STRATUM BREAKDOWN AFTER S3
**Job:** 23821658 | `run_eval.sh` → `eval_full_analysis.py` on Stage 3 best checkpoint  
**Result:** Per-stratum T1 mAP after Stage 3 (gradient starvation signature):

```
tiny=0.0004  small=0.0670  normal=0.0878  large=0.0000
```

Large-target mAP collapses from 0.461 (after Stage 1) to **0.000** — reported in Table 4.

---

## Scale-Stratified Herding Buffer

### `build_buffers_23391452.out` ✅ BUFFER CONSTRUCTION
**Job:** 23391452 | `run_build_buffers.sh` → `build_herding_buffer.py`  
**Checkpoint:** Stage 2 seed-42 best.pt on Anti-UAV-RGBT val (60,620 UAV-present frames)

```
Stratum     Available   Selected
tiny              197         75   (38.1% — tiny is underrepresented)
small          14,798         75
normal         42,608         75
large           3,017         75
TOTAL          60,620        300
```

Both herding and random-stratified variants built. Herding buffer at `/projects/prjs2041/runs/stage2/seed42/herding_buffer.pt`.

---

## Stage 3 — Herding Attempts (budget exhaustion)

All herding jobs were cancelled by Snellius before convergence. The replay loss was active and the training loop was functional, but compute budget ran out.

### `stage3_herding_23425651.err` — Attempt 1
**Job:** 23425651 | Cancelled at epoch 0 during DDP init issue.

### `stage3_herding_23443056.err` — Attempt 3
**Job:** 23443056 | Reached epoch 8 before wall-time cancellation. Replay loss decaying but T1 still near zero.

### `stage3_herding_23449442.err` — Attempt 4 / Final
**Job:** 23449442 | Last attempt before total budget exhaustion. Cancelled at epoch 1.

```
Epoch 0: cst=0.077  replay=0.148  T1 mAP=0.008
[2026-06-04T01:44:34] JOB 23449442 CANCELLED DUE TO SIGNAL Terminated
```

Experimental comparison of SSH vs naive fine-tuning is left as future work.

---

## Tracking Evaluation

### `track_cst_23824125.out` ✅ SOT TRACKING ON CST
**Job:** 23824125 | `run_tracking_eval.sh` → `eval_tracking_cst.py`  
**Checkpoint:** Stage 3 naive best (epoch 3) on CST val (40 sequences, 39,055 frames)

```
SR@0.5  = 0.0119   (Success Rate, IoU≥0.5)
PR@20   = 0.2189   (Precision, ≤20px centre error)
IDSW    = 135      (identity switches across 40 sequences)
```

Confirms near-total loss of tracking ability on T3 data after Stage 3 naive training.

---

## Legacy Logs (early development, kept for traceability)

| File | Job | Notes |
|------|-----|-------|
| `stage1_ddp_22522864.out` | 22522864 | First complete S1 — mAP=0.6617 at ep. 54. Superseded by 23297129. |
| `stage2_22637974.out` | 22637974 | Early S2 dev run, single GPU. Superseded. |
| `eval_stage1_22583396.out` | 22583396 | Full eval of 22522864 checkpoint. |
| `stage1_ddp_22487080.out` | 22487080 | Failed S1 DDP run (dataset debugging). |
| `stage1_ddp_22501402.out` | 22501402 | Failed S1 DDP run. |
| `stage1_ddp_22501881.out` | 22501881 | Failed S1 DDP run. |
| `stage1_ddp_22505545.out` | 22505545 | Failed S1 DDP run. |
| `stage1_ddp_22519342.out` | 22519342 | Failed S1 DDP run. |
