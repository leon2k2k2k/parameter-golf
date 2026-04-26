# Spec 046J — per-block adaptive SDClip

**Slug:** `046J-perblock-sdclip`
**Created:** 2026-04-27
**Status:** DRAFT — needs ~30-50 lines code (per-block clip env var)
**Branch:** `exp/046-quant-repair`
**Commit:** TBD
**Parent:** `research/ideas/quant-repair.md` (Q-N), 046G results

## Hypothesis

Currently SDClip is per-CATEGORY (MLP=12.0, ATTN=13.0, EMBED=15.0) — uniform
across all 11 blocks within a category. But blocks have different weight
statistics:
- Early layers process raw embeddings
- Middle layers do most of the heavy attention/feature mixing
- Late layers project toward output

PR #1689 showed per-tensor adaptive clip via Hessian sensitivity gives
-0.001 to -0.003 BPB. We never ported that to our stack. Here we test the
simpler version: **per-block clip values via grid search**, not Hessian-derived.

If different blocks have different optima, we can find per-block clips that
beat 046G's uniform tightening AT BOTH BPB AND BYTES (some blocks may want
LOOSER, some TIGHTER).

## Code change required

Add new env vars `MLP_CLIP_SIGMAS_PER_BLOCK` and `ATTN_CLIP_SIGMAS_PER_BLOCK`
that take comma-separated arrays:
```bash
MLP_CLIP_SIGMAS_PER_BLOCK="11.0,11.5,12.0,12.0,11.5,12.0,12.0,12.0,11.5,11.0,10.5"
```

In `gptq_mixed_quantize()`, when picking clip sigma for a tensor named
`blocks.N.mlp.fc.weight`, look up the per-block value if the env var is set,
else fall back to per-category default.

~30-50 lines code.

## Arms

### Phase 1 — sensitivity analysis (3 arms)

Find which blocks benefit from tighter vs looser:

| Arm | MLP per-block | ATTN per-block |
|---|---|---|
| **046J-mlp-block-sweep-A** | tighter early (11.0,11.0,11.0,12,12,12,12,12,12,12,12) | baseline |
| **046J-mlp-block-sweep-B** | tighter middle (12,12,12,11.0,11.0,11.0,12,12,12,12,12) | baseline |
| **046J-mlp-block-sweep-C** | tighter late (12,12,12,12,12,12,12,12,11.0,11.0,11.0) | baseline |

### Phase 2 — refined per-block based on Phase 1 results

If e.g. early-tight wins big and middle-tight loses, sweep deeper:

| Arm | Strategy |
|---|---|
| **046J-targeted** | tighter on winning blocks, looser on losing blocks; pick to fit cap |
| **046J-mixed-bestof-each** | best per-block clip from Phase 1 sweep, combined |

### Phase 3 — combine with 046G learnings

| Arm | Strategy |
|---|---|
| **046J-perblock-tighter** | per-block sweep around 046G-tighter values |

Cost: Phase 1 ~$3, Phase 2 ~$2, Phase 3 ~$2. Total ~$7.

## Predicted outcome

If per-block adaptive helps:
- -0.0005 to -0.001 BPB beyond 046G's uniform tightening
- AT THE SAME OR LOWER artifact size (because losing blocks get looser, recouping bytes)

If null (all blocks have similar optima):
- Confirms per-CATEGORY clip was already near-optimal granularity
- Direction closes

## Acceptance

Reference = 046 verification (1.07467, 15,953,718 bytes).

Per arm:
- **Strong legal win**: quantized < 1.0735 AND size ≤ 16,000,000
- **Legal win**: quantized < 1.0739 AND size ≤ 16,000,000
- **Pareto match**: quantized 1.0739–1.0760 AND size ≤ 16,000,000

## Risk

Low — same SDClip primitive, just finer per-block control. Could overflow cap
if all blocks tighten; mitigated by setting some blocks looser.

## Why this might find something new

046G tested uniform tightening by (-0.5σ, -1.0σ, -1.5σ). The "winner" was
-1.5σ with -0.00216 BPB but +689 KB over cap. Per-block adaptive could give
the same -0.002 BPB at FEWER bytes if heterogeneity exists across blocks.

This is the only Tier-1 quant idea where the "byte cost" can be amortized
non-uniformly across the model.

## Cost

~$7, ~1 hour wallclock.
