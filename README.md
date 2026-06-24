# Catastrophic Forgetting in Sequential Thermal Anti-UAV Detection
### Scale-Conditioned Gradient Imbalance as a Candidate Forgetting Mechanism

**Author:** Khac Duc Giang Nguyen (16265858)  
**Supervisor:** Dr. Seyed Sahand Mohammadi Ziabari  
**Institution:** University of Amsterdam — MSc Information Studies (Data Science)  
**Submitted:** June 2026

---

## What This Thesis Is About

Thermal infrared UAV detectors must remain accurate as operational datasets evolve. Sequential fine-tuning on a new dataset destroys performance on previous ones — a phenomenon called catastrophic forgetting. This thesis trains YOLOMG through a three-stage curriculum of increasing scale difficulty, measures precisely *when*, *how much*, and *why* forgetting happens, and proposes Scale-Stratified Herding as a targeted remedy.

**Core finding:** Scale-distribution shift (Stage 3, CST Anti-UAV: 97.7% tiny/small targets, 0% large) causes **18× more forgetting** than cross-domain adaptation (Stage 2, Anti-UAV410). The mechanism is **gradient starvation** — learnable-parameter cosine similarity S2→S3 is 0.987 (weights barely moved) yet T1 mAP collapses from 0.640 to 0.068. Large-target detection reaches 0.000 mAP within the first training epoch. BN running statistics drift 2× more than gradient-updated weights (L2-rel ≈ 0.24 vs 0.12), confirming that the network re-estimates normalisation stats for CST's tiny-target distribution while conv weights remain largely unchanged.

---

## Research Questions

| | Question | Finding |
|---|---|---|
| **H1** | Is catastrophic forgetting measurable under the three-stage curriculum? | FM = −0.605; scale shift drives 18× more forgetting than domain shift |
| **RQ1** | Does Knowledge Distillation preserve T1 performance during T2 domain shift? | FM = −0.033 ± 0.004 across three seeds; 95% T1 retention |
| **RQ2** | Does scale-distribution shift characterise the forgetting pattern in Stage 3? | Large-target collapse to 0.000 mAP despite cosine sim 0.987; evidence points to scale-conditioned gradient imbalance |
| **RQ3** | Does Scale-Stratified Herding mitigate large-target forgetting? | Buffer built and integrated; empirical comparison left as future work |

---

## Three-Stage Curriculum

| Stage | Task | Dataset | Result |
|-------|------|---------|--------|
| 1 | Supervised baseline | Anti-UAV-RGBT (208,737 frames) | mAP@0.5 = **0.6725** |
| 2 | KD fine-tuning | Anti-UAV410 (438,397 frames) | FM = **−0.033 ± 0.004** (3 seeds) |
| 3 | Naive baseline | CST Anti-UAV (245,471 frames) | FM = **−0.605**, best at epoch 3 |
| 3 | Scale-Stratified Herding | CST Anti-UAV | Cancelled — compute budget exhausted |

---

## Key Results

### Stage 1 — T1 Baseline

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.6725** |
| Best epoch | 49 / 100 |
| Tiny-stratum mAP | 0.009 |
| Small-stratum mAP | 0.579 |
| Normal-stratum mAP | 0.719 |
| Large-stratum mAP | 0.461 |
| Training | ~40 h (4×A100, Snellius) |

### Stage 2 — Knowledge Distillation (3 seeds: 42, 123, 999)

Loss: `L_total = L_det + λ_kd × L_kd`, where `L_kd` = MSE between student and frozen teacher at P3/P4/P5. `λ_kd = 1.0`.

| Metric | Value |
|--------|-------|
| T2 mAP@0.5 (Anti-UAV410) | **0.428 ± 0.002** |
| T1 mAP@0.5 retained | 0.640 |
| Forgetting Measure FM | **−0.033 ± 0.004** |
| T1 retention | **95%** |
| Mean cosine sim S1→S2 (all-params) | **0.911** (per-seed: 0.917 / 0.921 / 0.896) |
| Mean cosine sim S1→S2 (learnable-only) | **0.935** (seed 42) |
| Best T2 epoch (seed 42) | 16 / 32 |

<details>
<summary>Epoch-level breakdown (seed 42, 32 epochs)</summary>

| Epoch | L_det | L_kd | kd/det | T2 mAP | FM |
|-------|-------|------|--------|--------|----|
| 0 | 0.0580 | 0.0791 | 1.36 | 0.403 | −0.007 |
| 5 | 0.0355 | 0.0768 | 2.16 | 0.419 | −0.035 |
| 10 | 0.0317 | 0.0739 | 2.33 | 0.406 | −0.046 |
| 15 | 0.0293 | 0.0712 | 2.43 | 0.417 | −0.042 |
| **16** | **0.0289** | **0.0705** | **2.44** | **0.426** | **−0.037** |
| 20 | 0.0273 | 0.0677 | 2.48 | 0.419 | −0.041 |
| 25 | 0.0256 | 0.0648 | 2.54 | 0.415 | −0.044 |
| 31 | 0.0233 | 0.0612 | 2.62 | 0.422 | −0.037 |

</details>

### Stage 3 — Naive Baseline (no replay, 19 epochs)

| Metric | Value |
|--------|-------|
| FM (best checkpoint, ep. 3) | **−0.605** |
| T1 mAP@0.5 at ep. 3 | 0.068 |
| T1 mAP@0.5 at ep. 18 | 0.017 |
| Large-stratum T1 mAP | **0.000** (from 0.461 after Stage 1) |
| Normal-stratum T1 mAP | 0.079 (from 0.719) |
| Small-stratum T1 mAP | 0.060 |
| Mean cosine sim S2→S3 (all-params) | **0.967** |
| Mean cosine sim S2→S3 (learnable-only) | **0.987** |
| BN running-stat drift (L2-rel) | **0.24** vs learnable weights 0.12 |
| P5 head gradient ratio RGBT/CST | **1.7×** (0.654 vs 0.387) |

**Gradient starvation signature:** learnable-parameter cosine similarity S2→S3 is 0.987 — weights barely moved — yet forgetting is 18× worse than Stage 2. Large-target features received zero gradient signal because CST has 0% large targets. BN running statistics drift 2× more than conv weights, indicating the network adapts its normalisation to CST's tiny-target distribution while the gradient-updated parameters remain largely frozen.

---

## Scale-Stratified Herding (Design Contribution)

Buffer built and verified using Stage 2 seed-42 checkpoint on Anti-UAV-RGBT val (60,620 UAV-present frames). Experimental comparison against naive fine-tuning is left as future work.

| Stratum | Eligible frames | Selected | Sampling rate |
|---------|----------------|----------|---------------|
| Tiny (<16 px) | 197 | 75 | 38.1% |
| Small (16–32 px) | 14,798 | 75 | 0.5% |
| Normal (32–64 px) | 42,608 | 75 | 0.2% |
| Large (>64 px) | 3,017 | 75 | 2.5% |
| **Total** | 60,620 | **300** | — |

---

## Model: YOLOMG

Dual-input YOLOv5-based detector (Guo et al., 2025). YOLOMG was selected because the original thesis scope included motion-based drone detection using the mask32 channel. As the scope narrowed to a continual learning protocol, the motion channel was fixed to zeros throughout — Anti-UAV-RGBT and Anti-UAV410 provide insufficient inter-frame motion signal at the acquisition distances and frame rates available, and pre-computing motion masks for three large datasets was not feasible within the compute budget.

- `img1`: IR appearance frame
- `img2`: zeros (motion channel — unused in this study)
- 318 layers, 335 named parameter tensors, 640×512 input
- Config: `YOLOMG-main/models/dual_uav2.yaml`

---

## Repository Structure

```
YOLOMG-main/                     YOLOMG detector source (Guo et al., 2025)
src/
  datasets/                      Dataset loaders
    antiuav_rgbt.py              Anti-UAV-RGBT (video, cv2.VideoCapture)
    antiuav410.py                Anti-UAV410 (JPEG frames)
    cst.py                       CST Anti-UAV (JPEG frames)
    ard100.py                    ARD100 (motion mask pre-computation only)
    base.py                      BaseUAVDataset
  train_stage1.py                Stage 1 supervised training (DDP)
  train_stage2.py / _ddp.py      Stage 2 KD fine-tuning (single-GPU / DDP)
  train_stage3.py                Stage 3 naive baseline + herding replay
  build_herding_buffer.py        Scale-Stratified Herding buffer construction
  eval_full_analysis.py          Per-stratum mAP, PR curves, per-seq mAP
  eval_stage1.py                 Stage 1 checkpoint evaluation
  eval_tracking_cst.py           SOT tracking eval on CST (SR@0.5, PR@20, IDSW)
  visualise_detections_rgbt.py   Bounding-box overlay frames
  parameter_drift.py             Inter-stage cosine similarity analysis
  compute_drift_s2s3.py          S2→S3 parameter drift (layer-group breakdown)
  plot_training_analysis.py      Training curve plots
  plot_scale_distribution.py     Scale distribution bar charts
  plot_multirun_ci.py            Multi-seed CI plots
  audit_datasets.py              Dataset sanity checks
  test_datasets.py               Dataset loader unit tests
  json2yolo.py                   Annotation format conversion
  grad_starvation_diagnostic.py  Gradient norm diagnostic per head (P3/P4/P5)
  eval_stratum_t1.py             Per-stratum T1 mAP across all three stages
  parameter_drift.py             Inter-stage cosine similarity (--learnable-only flag)
  run_stage1_ddp.sh              SLURM: Stage 1 (4×A100, 72 h)
  run_stage1.sh                  SLURM: Stage 1 single-GPU
  run_stage1_rerun.sh            SLURM: Stage 1 rerun from checkpoint
  run_stage2_ddp.sh              SLURM: Stage 2 DDP (4×H100)
  run_stage2.sh                  SLURM: Stage 2 single-GPU
  run_stage3_naive.sh            SLURM: Stage 3 naive baseline
  run_stage3_herding.sh          SLURM: Stage 3 herding (cancelled)
  run_stage3_random_stratified.sh SLURM: Stage 3 random-stratified replay
  run_eval.sh                    SLURM: full evaluation
  run_eval_pr_s3.sh              SLURM: PR curve evaluation Stage 3
  run_tracking_eval.sh           SLURM: tracking evaluation
  run_vis.sh                     SLURM: detection visualisation
  run_analysis_drift.sh          SLURM: parameter drift analysis (all-params)
  run_drift_s2s3.sh              SLURM: S2→S3 drift analysis
  run_drift_s2s3_learnable.sh    SLURM: S2→S3 learnable-only drift
  run_grad_diagnostic.sh         SLURM: gradient starvation diagnostic
  run_eval_stratum_t1.sh         SLURM: per-stratum T1 evaluation
  run_analysis_stage2.sh         SLURM: Stage 2 multi-seed aggregation
  run_build_buffers.sh           SLURM: herding buffer construction
  run_data_pipeline.sh           SLURM: full data extraction pipeline
  run_generate_masks.sh          SLURM: motion mask generation
analysis/                        Per-layer drift CSVs and figures (see analysis/)
logs/                            SLURM output logs (see logs/README.md)
docs/
  stage2_progress.png            Stage 2 training curves
src/figures/                     Generated paper figures
  fig_size_distribution.png/pdf  Scale distribution across datasets
  fig_frame_counts.png/pdf       Frame counts per dataset split
  fig_visibility.png/pdf         Target visibility rates
```

---

## Installation (Snellius HPC)

```bash
module load 2023
module load Miniconda3/23.5.2-0
conda create -n uav_master python=3.9
conda activate uav_master
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless numpy==2.0.2 scipy pandas pyyaml tqdm
```

Always use the absolute interpreter path on Snellius:
```bash
/home/knguyen1/.conda/envs/uav_master/bin/python script.py
```

---

## Implementation Notes

**SLURM GPU syntax:** `--gres=gpu:a100:1` or `--gres=gpu:h100:1`. A100 ≈ 128 SBU/GPU-hour; H100 ≈ 192 SBU/GPU-hour (768 SBU/wall-hour for a 4-GPU job).

**DDP:** `torchrun --standalone --nproc_per_node=4`, NCCL backend. Use `find_unused_parameters=True` — the zero motion channel leaves `backbone1` without gradients.

**CSTDataset:** `IR_label.json` uses key `gt` (not `gt_rect`); images at `{seq}/{frame:06d}.jpg` (1-based, 6-digit).

**Size bins (UAV-specific):** tiny <16 px, small 16–32 px, normal 32–64 px, large ≥64 px (longest bbox side). Standard COCO bins (32/96/192 px) are inappropriate for sub-100 px thermal UAV targets.

**VideoCapture:** `persistent_workers=False` — with `True`, workers accumulate RAM and kill long jobs (observed at Stage 1 epoch 83; best checkpoint at epoch 49 was safe).

**ap_per_class (YOLOMG-specific):** returns 7 values; `correct` array must be shape `(N, num_iou_thresholds)`.

**Total compute used:** ~113,140 SBU of a 120,000 SBU allocation (budget exhausted after Stage 3 herding crash-fix cycles).
