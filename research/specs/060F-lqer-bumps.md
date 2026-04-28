# Spec 060F — LQER capacity bumps on 060A baseline (eval-only)

**Date:** 2026-04-29
**Branch:** `research` (config-only)
**Parent:** 060A `final_model.pt`.

## Hypothesis

046D measured LQER `RANK=6` and `TOP_K=5/8` as null on 045-armD. BUT — 046D ran without lrzip headroom and on a different base. With #1855's lrzip compressor giving ~280 KB headroom, we can afford bigger LQER tensors. Per 046K's rank-adaptive analysis, the marginal residual on additional rank is small but might compound with other 060 levers.

## Baseline

060A.

## Expected Δ

**−0.001 to −0.003 BPB**, low confidence (046D measured null on prior base).

## Accept criteria

- val_bpb ≤ (060A − 0.0005)
- artifact ≤ 16 MB

## Config diff vs 060A

Three arms (sequential, pick best):

**Arm F1 — RANK bump:**
```
LQER_RANK: 4 → 5     (cost ~50 KB)
LQER_TOP_K: 3        (unchanged)
```

**Arm F2 — TOP_K bump:**
```
LQER_RANK: 4         (unchanged)
LQER_TOP_K: 3 → 4    (cost ~30-50 KB)
```

**Arm F3 — finer asym groups:**
```
LQER_ASYM_GROUP: 64 → 32   (cost ~50-80 KB; finer per-group quant scales)
```

## Code changes

None. Env-var override + RESUME_FROM_CKPT.

## Hardware ladder

- 4×H100, eval-only.
- ~5-7 min wall per arm, ~$1-2 each.

## Seed plan

1 seed (42) per arm.

## Cost estimate

~$5-6 total for all three arms.

## Stop-early criteria

- Artifact > 16 MB → fail this arm
- 046D showed null on RANK=6; if F1 (RANK=5) is null, skip F2/F3

## Run command (per arm)

```bash
# Arm F1
SEED=42 RUN_LABEL=seed_42_F1 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
  LQER_RANK=5 \
  bash tmp_exec/launch_060_eval.sh
```
