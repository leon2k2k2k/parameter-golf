# Spec 060E — Full quant-repair stack on 060A (eval-only)

**Date:** 2026-04-29
**Branch:** `exp/060C-deploy-repair` (uses 060C's code port)
**Parent:** 060A pt + 060C code; depends on 060B and 060C results being individually positive.

## Hypothesis

Combine the two best validated levers:
- 060B's ATTN clip tightening (−0.0008)
- 060C's deploy-time quant repair (−0.001 to −0.005)

Expected combined gain: **−0.002 to −0.005 BPB**, assuming additivity (they target different error sources: SDClip reduces quant noise at calibration; deploy-time repair recovers passthrough fp16 params at eval).

## Baseline

060A.

## Expected Δ

**−0.002 to −0.005 BPB** vs 060A. Lower confidence on additivity — they could partially overlap.

## Accept criteria

- val_bpb ≤ min(060B, 060C) − 0.0005 (better than either alone)
- artifact ≤ 16 MB
- no instability

## Config diff vs 060A

```
ATTN_CLIP_SIGMAS=12.5            (from 060B)
DEPLOY_TIME_REPAIR_ENABLED=1     (from 060C)
DEPLOY_TIME_REPAIR_BATCHES=8
DEPLOY_TIME_REPAIR_SEQ_LEN=512
DEPLOY_TIME_REPAIR_LR=1e-3
DEPLOY_TIME_REPAIR_ITERS=5
```

## Code changes

Same as 060C (deploy-time repair port). 060B is config-only on top.

## Hardware ladder

- 4×H100, eval-only via RESUME_FROM_CKPT
- ~15 min wall, ~$3

## Stop-early criteria

- Either component alone (B or C) underperforms 060A → stop family; pivot to single-lever approach
- Combined val_bpb ≥ 060A → no synergy, treat as null and ship best-of-{B,C}

## Run command

```bash
SEED=42 RUN_LABEL=seed_42_full \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
  ATTN_CLIP_SIGMAS=12.5 \
  DEPLOY_TIME_REPAIR_ENABLED=1 DEPLOY_TIME_REPAIR_ITERS=5 \
  bash tmp_exec/launch_060_eval.sh
```
