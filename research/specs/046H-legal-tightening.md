# Spec 046H — legal-cap tightening (EMBED_BITS=6 to fund SDClip wins)

**Slug:** `046H-legal-tightening`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/046-quant-repair`
**Commit:** `a1bef8d` (no new code)
**Parent:** spec 046G (which won big but overflowed cap)

## Hypothesis

046G proved SDClip tightening is **monotonic and strong** (-0.00216 BPB at -1.5σ),
but ALL winning arms overflowed the **16,000,000 byte decimal cap**:

| Arm | Quantized | Δ | Size | Over cap? |
|---|---|---|---|---|
| baseline | 1.07467 | — | 15,953,718 | ✓ legal (46 KB headroom) |
| 046B-tight (-0.5σ) | 1.07381 | -0.00086 | 16,183,797 | ❌ +184 KB |
| 046G-tighter (-1.0σ) | 1.07322 | -0.00146 | 16,427,706 | ❌ +428 KB |
| 046G-tightest (-1.5σ) | 1.07251 | -0.00216 | 16,688,654 | ❌ +689 KB |
| 046G-mlp-only-tighter | 1.07379 | -0.00088 | 16,254,785 | ❌ +255 KB |
| 046G-attn-only-tighter | 1.07432 | -0.00035 | 16,078,450 | ❌ +78 KB |

**Strategy**: drop `EMBED_BITS=7 → 6` to free **~250-300 KB** of artifact, then
re-apply SDClip tightening within the legal budget.

EMBED is the largest tensor (8192 × 512 ≈ 4M params). Going int7→int6 saves
~14% of the embedding's quant bytes ≈ **~250-300 KB free**. LQER asym rank-4
on the embedding may absorb most of the precision loss.

## Predicted outcome

If `EMBED_BITS=6` alone doesn't regress more than the byte savings buy back
in tightening room, we can ship 046G's winning configs legally.

- **Optimistic**: emb6 costs +0.0005 (LQER absorbs most), tightening recovers -0.002 → net -0.0015 legal win
- **Realistic**: emb6 costs +0.001-0.002 (some emb damage), tightening recovers -0.0015 → net -0.0005 to neutral
- **Pessimistic**: emb6 costs +0.003+, no net win

## Arms

All resume from 045 armD checkpoint via RESUME_FROM_CKPT. Each arm sets
`EMBED_BITS=6` plus an SDClip variant. Run sequentially.

| Arm | MLP | ATTN | EMBED_CLIP | EMBED_BITS | Notes |
|---|---|---|---|---|---|
| **046H-emb6-baseline** | 12.0 | 13.0 | 15.0 | **6** | sanity: emb6 cost on plain config |
| **046H-emb6-tight** | 11.5 | 12.5 | 14.5 | **6** | 046B-tight + emb6 |
| **046H-emb6-tighter** | 11.0 | 12.0 | 14.0 | **6** | 046G-tighter + emb6 |
| **046H-emb6-tightest** | 10.5 | 11.5 | 13.5 | **6** | 046G-tightest + emb6 |
| **046H-emb6-mlp-only-tighter** | 11.0 | 13.0 | 15.0 | **6** | mlp-only + emb6 (cheapest tightening) |
| **046H-emb6-mlp-tightest** | 10.5 | 13.0 | 15.0 | **6** | push MLP only, hold attn/emb |
| **046H-emb6-clip-emb-tighter** | 12.0 | 13.0 | 14.0 | **6** | tighter clip on emb to compensate for bit loss |

7 arms, ~$7, ~50 min wallclock sequential.

## Config diff (per arm — example)

```bash
# 046H-emb6-baseline
EMBED_BITS=6
MATRIX_CLIP_SIGMAS=12.85
MLP_CLIP_SIGMAS=12.0
ATTN_CLIP_SIGMAS=13.0
EMBED_CLIP_SIGMAS=15.0
RUN_ID="046H-emb6-baseline"
```

All other env vars identical to 046 verification. RESUME_FROM_CKPT same path.

## Acceptance

Reference = 046 verification quantized = 1.07467, **size 15,953,718**.

Per arm:
- **Strong legal win**: quantized < 1.0735 AND size ≤ 16,000,000
- **Legal win**: quantized < 1.0739 AND size ≤ 16,000,000
- **Pareto plateau**: quantized 1.0739–1.0760 AND size ≤ 16,000,000 (technically legal but not interesting)
- **Illegal**: size > 16,000,000 (any quant value irrelevant — won't ship)
- **Kill**: quantized > 1.0760 (emb6 hurt too much OR config off Pareto)

## What to watch

- **`Total submission size quantized+brotli:`** in log — MUST be ≤ 16,000,000
- Quantized val_bpb
- Whether emb6 alone (`046H-emb6-baseline`) costs ≥ +0.001 — if yes, the budget freed isn't worth the precision loss

## Decision tree

| Best legal arm | Next |
|---|---|
| Wins ≥ −0.001 (legal) | Lock as new baseline; sweep further (LQER tweaks, deeper tightening with the freed budget) |
| Wins 0 to −0.001 (legal) | Marginal; combine with EMA decay sweep + multi-seed final |
| All emb6 arms regress | EMB6 too lossy on this stack; need different byte-saving (LQER_TOP_K=2 or MATRIX_BITS=5 mixed) |

## Cost

~$7 (7 arms × $1).
