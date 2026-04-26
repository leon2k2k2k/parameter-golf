# Spec 046G — SDClip tightening follow-up (after 046B win)

**Slug:** `046G-sdclip-tighter-followup`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/046-quant-repair`
**Commit:** `031d675` (no new code; config-only arms)
**Parent:** spec 046B (SDClip sigma re-tune), idea Q1 in `research/ideas/quant-repair.md`

## Hypothesis

046B established a clear directional signal:

| 046B arm | Quantized | Δ vs 046 baseline (1.07467) |
|---|---|---|
| sdclip-tight (MLP=11.5, ATTN=12.5, EMBED=14.5) | **1.07381** | **−0.00086** ✓ win |
| sdclip-mlp-tight (MLP=11.5 only) | 1.07422 | −0.00045 ✓ small win |
| sdclip-loose (MLP=12.5, ATTN=13.5, EMBED=15.5) | 1.07499 | +0.00031 (noise) |
| sdclip-loose (rerun) | 1.07538 | +0.00071 |

**Tighter SDClip wins, looser hurts.** Tight at -0.5σ gave -0.00086. The mlp-only
arm got half that (-0.00045), so attn + embed contribute roughly the other half.

This spec pushes the lever further (tighter still) and localizes the signal
across the three categories.

## Predicted outcome

- If monotonic: tighter still gives −0.0015 to −0.002 BPB
- If past Pareto: regress (clip too tight → too many values dropped from quant)
- Localization arms tell us where to focus future tuning

## Constraints

- **Size budget**: 046B-tight hit 16.18 MB submission size. Cap is 16,777,216 bytes (16 MiB) = ~600 KB headroom. Each -0.5σ notch tightens by ~50-100 KB. Need to watch size on aggressive arms.
- **GPTQ noise floor**: 046B-loose vs loose-rerun spread of ~0.0004 sets noise. Anything under -0.0010 is real signal.

## Arms

All resume from 045 armD checkpoint, RESUME_FROM_CKPT mode. Run sequentially.

### Push-the-lever arms

| Arm | MLP | ATTN | EMBED | Predicted |
|---|---|---|---|---|
| **046G-tighter** | 11.0 | 12.0 | 14.0 | extends 046B-tight by another -0.5σ |
| **046G-tightest** | 10.5 | 11.5 | 13.5 | aggressive; risk of size overflow + Pareto regression |

### Localization arms (helpful to attribute the gain)

| Arm | MLP | ATTN | EMBED | Tests |
|---|---|---|---|---|
| **046G-mlp-only-tighter** | 11.0 | 13.0 | 15.0 | MLP alone at -1.0σ |
| **046G-attn-only-tighter** | 12.0 | 12.0 | 15.0 | ATTN alone at -1.0σ |
| **046G-embed-only-tighter** | 12.0 | 13.0 | 14.0 | EMBED alone at -1.0σ |

### Invest-size-budget arms

| Arm | MLP | ATTN | EMBED | EMBED_BITS | Predicted |
|---|---|---|---|---|---|
| **046G-tight-emb8** | 11.5 | 12.5 | 14.5 | **8** (vs current 7) | more emb precision; ~+500KB size |

(Skip MATRIX_BITS=5/7 — PR #1646 ablation already showed int5 fails on this stack
and int7 would blow size budget by ~1.5 MB.)

## Config diff (per arm, vs 046 verification baseline)

```bash
# 046G-tighter
MLP_CLIP_SIGMAS=11.0
ATTN_CLIP_SIGMAS=12.0
EMBED_CLIP_SIGMAS=14.0
RUN_ID="046G-sdclip-tighter"

# 046G-tightest
MLP_CLIP_SIGMAS=10.5
ATTN_CLIP_SIGMAS=11.5
EMBED_CLIP_SIGMAS=13.5
RUN_ID="046G-sdclip-tightest"

# 046G-mlp-only-tighter
MLP_CLIP_SIGMAS=11.0
ATTN_CLIP_SIGMAS=13.0
EMBED_CLIP_SIGMAS=15.0
RUN_ID="046G-sdclip-mlp-only-tighter"

# 046G-attn-only-tighter
MLP_CLIP_SIGMAS=12.0
ATTN_CLIP_SIGMAS=12.0
EMBED_CLIP_SIGMAS=15.0
RUN_ID="046G-sdclip-attn-only-tighter"

# 046G-embed-only-tighter
MLP_CLIP_SIGMAS=12.0
ATTN_CLIP_SIGMAS=13.0
EMBED_CLIP_SIGMAS=14.0
RUN_ID="046G-sdclip-embed-only-tighter"

# 046G-tight-emb8
MLP_CLIP_SIGMAS=11.5
ATTN_CLIP_SIGMAS=12.5
EMBED_CLIP_SIGMAS=14.5
EMBED_BITS=8
RUN_ID="046G-sdclip-tight-emb8"
```

All other env vars identical to 046 verification (LOOP_ITER_EMBEDS=0, MLP_ONLY_FROM_PASS=0,
LOOP_SCALE_INIT=recip, RESUME_FROM_CKPT=...).

## Acceptance

Reference = 046 verification quantized = 1.07467.

Per arm:
- **Strong win**: < 1.0730 (−0.0017 from baseline)
- **Win**: < 1.0739 (−0.00077, beats 046B-tight)
- **Match 046B-tight**: 1.0738 ± 0.0003 (Pareto plateau)
- **Noise**: 1.0739–1.0760
- **Kill**: > 1.0760 OR submission size > 16,777,216 bytes (cap overflow)

## What to watch

- `Total submission size quantized+brotli:` in log — must stay < 16,777,216
- For **046G-tightest**: likely runs hot on size; may need to back off
- For **046G-tight-emb8**: confirms whether embedding precision is the next bottleneck
- Localization triplet should sum to ~the tighter-arm gain if effects are linear

## Decision tree

| Result | Next |
|---|---|
| 046G-tighter wins clean (≥ −0.001) | Sweep further (-0.5σ more); consider stacking with 046A winner |
| 046G-tighter ≈ 046B-tight (plateau) | Lever exhausted at -0.5σ; combine with other levers |
| 046G-tighter regresses | Pareto past — tight at -0.5σ was the optimum, lock 046B-tight as new baseline |
| 046G-tight-emb8 wins big | Embed precision is bottleneck; spec int8-emb at multiple clip values |
| Localization arms reveal one category dominant | Spec a deep sweep on that category alone |

## Cost

~6 arms × ~$1 = ~$6. Sequential on same NE-1 4×H100 pod. ~45 min wallclock.

## Bigger picture

046B-tight win (-0.00086) gets us to 1.07381 quantized. SOTA frontier #1797 is 1.06157,
so we're still ~0.012 short. SDClip lever alone won't close the gap, but every 0.001
helps and stacks with 046A (calib batches), 046D (LQER), 046E (post-quant fit).

If 046G shows another -0.0005 to -0.001 stacked on 046B-tight, total quant-side win
could approach -0.002, getting us to ~1.0726 quantized. Combined with frontier-tap-in
work, that closes maybe a third of the gap.
