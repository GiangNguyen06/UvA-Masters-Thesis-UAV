# Catastrophic Forgetting in Sequential Thermal Anti-UAV Detection
### Scale-Conditioned Gradient Imbalance as a Candidate Forgetting Mechanism

**Author:** Khac Duc Giang Nguyen (16265858)  
**Supervisor:** Dr. Seyed Sahand Mohammadi Ziabari  
**Institution:** University of Amsterdam — MSc Information Studies (Data Science)  
**Submitted:** June 2026

---

## What This Research Is About

Thermal infrared UAV detectors must remain accurate as operational datasets evolve. Sequential fine-tuning on a new dataset destroys performance on previous ones — a phenomenon called catastrophic forgetting. This research trains YOLOMG through a three-stage curriculum of increasing scale difficulty, measures precisely *when*, *how much*, and *why* forgetting happens, and proposes Scale-Stratified Herding as a targeted remedy.

**Core finding:** Scale-distribution shift (Stage 3, CST Anti-UAV: 97.7% tiny/small targets, 0% large) causes **18× more forgetting** than cross-domain adaptation (Stage 2, Anti-UAV410). Inter-stage cosine similarity is 0.967 over all parameters (0.987 over gradient-updated weights alone) — weights barely moved — yet T1 mAP collapses from 0.640 to 0.068. Large-target detection reaches 0.000 mAP within the first training epoch. The evidence is consistent with a **scale-conditioned gradient imbalance** — a refinement of gradient starvation — as a candidate forgetting mechanism: the large stratum receives no positive gradient signal from CST's distribution, while BatchNorm running statistics drift roughly 2× more than the trained weights (L2-rel ≈ 0.24 vs 0.12), suggesting the network adapts its normalisation to the new distribution while the gradient-updated parameters remain largely unchanged.

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
| 3 | Scale-Stratified Herding | CST Anti-UAV | Buffer built and integrated; empirical comparison left as future work |

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

**Candidate mechanism — scale-conditioned gradient imbalance:** learnable-parameter cosine similarity S2→S3 is 0.987 — weights barely moved — yet forgetting is 18× worse than Stage 2. Large-target features received no positive gradient signal from CST's distribution (0% large targets). BN running statistics drift 2× more than conv weights, suggesting the network adapts its normalisation to CST's tiny-target distribution while gradient-updated parameters remain largely unchanged. The P5 head gradient ratio RGBT/CST = 1.7× provides direct evidence of large-target gradient starvation, though a definitive causal mechanism remains tentative.

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
YOLOMG-main/                      YOLOMG detector source (Guo et al., 2025)

src/
  datasets/
    antiuav_rgbt.py               Anti-UAV-RGBT loader (video, cv2.VideoCapture)
    antiuav_rgbt_frames.py        Anti-UAV-RGBT loader (pre-extracted JPEG frames)
    antiuav410.py                 Anti-UAV410 loader (JPEG frames)
    cst.py                        CST Anti-UAV loader (JPEG frames)
    ard100.py                     ARD100 loader (motion mask pre-computation only)
    base.py                       BaseUAVDataset shared interface

  Training
    train_stage1.py               Stage 1 supervised training on Anti-UAV-RGBT (DDP)
    train_stage2.py               Stage 2 KD fine-tuning on Anti-UAV410 (single-GPU)
    train_stage2_ddp.py           Stage 2 KD fine-tuning (DDP, 4×H100)
    train_stage3.py               Stage 3 naive fine-tuning + herding replay on CST

  Evaluation
    eval_full_analysis.py         Per-stratum mAP, PR curves, per-sequence mAP
    eval_stage1.py                Stage 1 checkpoint evaluation on Anti-UAV-RGBT
    eval_stratum_t1.py            Per-stratum T1 mAP evaluated after each stage
    eval_tracking_cst.py          SOT tracking eval on CST (SR@0.5, PR@20, IDSW)

  Analysis
    parameter_drift.py            Inter-stage cosine similarity; --learnable-only flag
                                  separates gradient-updated weights from BN buffers
    compute_drift_s2s3.py         S2→S3 parameter drift with per layer-group breakdown
    grad_starvation_diagnostic.py Per-head (P3/P4/P5) gradient L2-norm probe at the
                                  S2→S3 boundary; compares CST vs RGBT distributions
    build_herding_buffer.py       Scale-Stratified Herding buffer construction

  Preprocessing
    extract_antiuav_rgbt.py       Extract JPEG frames from Anti-UAV-RGBT .mp4 videos
    extract_ard100.py             Extract JPEG frames from ARD100 videos
    extract_frames_antiuav_rgbt.py Alternate frame extractor with quality options
    extract_frames.sh             Shell wrapper for frame extraction
    generate_masks_npz.py         Pre-compute mask32 motion difference maps (NPZ)
    generate_mask5_fixed.py       Fixed FD5 mask generator (ECC-based GMC)
    prepare_ard100.py             ARD100 annotation conversion to YOLO format
    json2yolo.py                  Generic JSON → YOLO annotation converter

  Plotting
    plot_training_analysis.py     Training curve plots (loss, mAP, FM per epoch)
    plot_scale_distribution.py    Scale distribution bar charts across datasets
    plot_multirun_ci.py           Multi-seed confidence interval plots

  Utilities
    audit_datasets.py             Dataset integrity and annotation sanity checks
    test_datasets.py              Dataset loader unit tests
    visualise_detections_rgbt.py  Bounding-box overlay visualisation for RGBT frames
    visualise_detections.py       General-purpose bounding-box overlay visualisation

  SLURM scripts — training
    run_stage1_ddp.sh             Stage 1 (4×A100, 72 h)
    run_stage1.sh                 Stage 1 single-GPU
    run_stage1_rerun.sh           Stage 1 rerun/resume from checkpoint
    run_stage2_ddp.sh             Stage 2 DDP (4×H100)
    run_stage2.sh                 Stage 2 single-GPU
    run_stage3_naive.sh           Stage 3 naive baseline
    run_stage3_herding.sh         Stage 3 herding replay (cancelled at epoch 1)
    run_stage3_random_stratified.sh Stage 3 random-stratified replay baseline
    run_multirun_stage2.sh        Stage 2 multi-seed launcher (seeds 42/123/999)

  SLURM scripts — evaluation & analysis
    run_eval.sh                   Full per-stratum evaluation
    run_eval_stage1.sh            Stage 1 checkpoint evaluation
    run_eval_pr_s3.sh             PR curve evaluation for Stage 3
    run_eval_stratum_t1.sh        Per-stratum T1 mAP across all three stages
    run_tracking_eval.sh          SOT tracking evaluation on CST
    run_analysis_drift.sh         Parameter drift analysis (all-params, S1→S2)
    run_drift_s2s3.sh             Parameter drift analysis (all-params, S2→S3)
    run_drift_s2s3_learnable.sh   Parameter drift (learnable-only, S2→S3)
    run_drift_s1s2_learnable.sh   Parameter drift (learnable-only, S1→S2)
    run_grad_diagnostic.sh        Gradient starvation diagnostic at S2→S3 boundary
    run_analysis_stage1.sh        Stage 1 result aggregation and visualisation
    run_analysis_stage2.sh        Stage 2 multi-seed aggregation
    run_analysis_stage2_h100.sh   Stage 2 multi-seed aggregation (H100 partition)

  SLURM scripts — data pipeline
    run_data_pipeline.sh          Full data extraction pipeline
    run_generate_masks.sh         Motion mask (mask32) generation
    run_build_buffers.sh          SSH buffer construction
    run_vis.sh                    Detection visualisation
    run_vis_stage1.sh             Stage 1 detection visualisation
    run_vis_stage2.sh             Stage 2 detection visualisation

  figures/
    fig_size_distribution.png/pdf Scale distribution across datasets (Figure 1)
    fig_frame_counts.png/pdf      Annotated frame counts per dataset
    fig_visibility.png/pdf        Target visibility rates per dataset

analysis/
  drift_s1s2/                     S1→S2 drift: learnable-only CSVs + figures
  drift_s2s3/                     S2→S3 drift: all-params and learnable-only CSVs + figures
  diagnostics/grad_starvation/    Per-head gradient norm CSVs (seeds 42, 123, 999)

logs/                             SLURM output logs (see logs/README.md)
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
