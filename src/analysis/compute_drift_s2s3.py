#!/usr/bin/env python3
"""
compute_drift_s2s3.py
---------------------
Compute parameter drift (cosine similarity) between Stage 2 and Stage 3
YOLOMG checkpoints.

Usage (Snellius login node — CPU only, no GPU needed):
  python3 compute_drift_s2s3.py \
      --s2 /projects/prjs2041/runs/stage2/seed42/weights/best.pt \
      --s3 /projects/prjs2041/runs/stage3/naive2/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

# YOLOMG must be on sys.path so the model class can be unpickled
YOLOMG_ROOT = Path("/projects/prjs2041/YOLOMG")
if str(YOLOMG_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOMG_ROOT))

import numpy as np
import torch
import torch.nn.functional as F


def load_state_dict(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # YOLOMG checkpoints store EMA weights under 'model'
    model = ck.get("model", ck)
    if hasattr(model, "state_dict"):
        return model.float().state_dict()
    # Already a plain state dict
    return {k: v.float() for k, v in model.items()}


def compute_cosine_sims(sd_a: dict, sd_b: dict) -> list[float]:
    sims = []
    for k in sd_a:
        if k not in sd_b:
            continue
        va = sd_a[k].flatten()
        vb = sd_b[k].flatten()
        if va.numel() < 2:
            continue
        cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        sims.append(cos)
    return sims


def main():
    parser = argparse.ArgumentParser(description="Cosine similarity: S2 → S3 weights")
    parser.add_argument("--s2", default="/projects/prjs2041/runs/stage2/seed42/weights/best.pt",
                        help="Path to Stage 2 best.pt")
    parser.add_argument("--s3", default="/projects/prjs2041/runs/stage3/naive2/weights/best.pt",
                        help="Path to Stage 3 best.pt")
    args = parser.parse_args()

    print(f"Stage 2 : {args.s2}")
    print(f"Stage 3 : {args.s3}")
    print("Loading checkpoints ...")

    sd2 = load_state_dict(args.s2)
    sd3 = load_state_dict(args.s3)

    print(f"  S2 keys: {len(sd2)}  |  S3 keys: {len(sd3)}")

    sims = compute_cosine_sims(sd2, sd3)

    print()
    print("=" * 50)
    print("S2 → S3 Parameter Drift")
    print("=" * 50)
    print(f"  Parameters analysed : {len(sims)}")
    print(f"  Mean cosine sim     : {np.mean(sims):.4f}")
    print(f"  Median cosine sim   : {np.median(sims):.4f}")
    print(f"  Std                 : {np.std(sims):.4f}")
    print(f"  Min                 : {np.min(sims):.4f}")
    print(f"  Max                 : {np.max(sims):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
