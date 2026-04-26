# Spec 046H — EMBED_BITS sweep + legal-cap SDClip tightening

**Slug:** `046H-legal-tightening`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/046-quant-repair`
**Commit:** `78b8047` (no new code)
**Parent:** spec 046G results (proven SDClip lever, all illegal due to 16M cap)

## Hypothesis

046G proved SDClip tightening is **monotonic and strong** (-0.00216 BPB at -1.5σ),
but ALL winning arms overflow the **16,000,000 byte decimal cap**.

The token embedding (`tok_emb`, 8192 × 512 ≈ 4.2M params, ~3.7 MB after pruning+brotli)
is the **largest single tensor**. Lowering its bit-width frees substantial bytes:

| EMBED_BITS | Approx tok_emb storage | Δ vs current (7) | Headroom freed |
|---|---|---|---|
| 8 | ~4.0 MB | +250 KB | NONE (worse) |
| **7 (current)** | **~3.7 MB** | **0** | — |
| 6 | ~3.4 MB | -250 KB | enough for 046B-tight legal |
| 5 | ~3.1 MB | -500 KB | enough for 046G-tighter legal |

The val_bpb cost of int6/int5 emb is unknown on our stack. LQER asym rank-4 already
covers `tok_emb`, which should absorb some of the precision loss.

## Strategy

Three-stage sweep:
1. **Pure EMBED_BITS sweep** — measure raw cost of int6 vs int5 (3 arms)
2. **EMBED_BITS=5/6 + tighter EMBED_CLIP_SIGMAS** — clip retuning for fewer bits (2 arms)
3. **Stacked: EMBED_BITS=6/5 + 046B/G SDClip wins** — the actual legal-shipping candidates (6 arms)

## Arms

All resume from 045 armD checkpoint via RESUME_FROM_CKPT. Run sequentially.

### Stage 1 — pure EMBED_BITS cost

Holding all other quant config at baseline:

| Arm | EMBED_BITS | EMBED_CLIP_SIGMAS | Expected |
|---|---|---|---|
| **046H-emb6-pure** | 6 | 15.0 | small regression (LQER absorbs most) |
| **046H-emb5-pure** | 5 | 15.0 | larger regression |

### Stage 2 — EMBED clip retune for fewer bits

Tighter clip preserves more outliers when bit-width is tight:

| Arm | EMBED_BITS | EMBED_CLIP_SIGMAS | Expected |
|---|---|---|---|
| **046H-emb6-clip12** | 6 | 12.0 | tighter clip recovers some int6 loss |
| **046H-emb5-clip12** | 5 | 12.0 | tighter clip helps int5 more |
| **046H-emb5-clip10** | 5 | 10.0 | very tight, may push too far |

### Stage 3 — Stacked: low-bit emb + SDClip tightening

Combine the byte-saving with the proven SDClip wins:

| Arm | EMBED_BITS | MLP | ATTN | EMBED clip | Notes |
|---|---|---|---|---|---|
| **046H-emb6-tight** | 6 | 11.5 | 12.5 | 14.5 | 046B-tight + emb6 |
| **046H-emb6-tighter** | 6 | 11.0 | 12.0 | 14.0 | 046G-tighter + emb6 |
| **046H-emb6-tightest** | 6 | 10.5 | 11.5 | 13.5 | 046G-tightest + emb6 (max combo) |
| **046H-emb5-tight** | 5 | 11.5 | 12.5 | 14.5 | 046B-tight + emb5 |
| **046H-emb5-tighter** | 5 | 11.0 | 12.0 | 14.0 | 046G-tighter + emb5 |
| **046H-emb5-tightest** | 5 | 10.5 | 11.5 | 13.5 | 046G-tightest + emb5 |

**Total: 11 arms × ~$1 = ~$11, ~90 min sequential.**

## Config diff (per arm — example)

```bash
# 046H-emb6-tightest (likely best legal candidate)
EMBED_BITS=6
MATRIX_CLIP_SIGMAS=12.85
MLP_CLIP_SIGMAS=10.5
ATTN_CLIP_SIGMAS=11.5
EMBED_CLIP_SIGMAS=13.5
RUN_ID="046H-emb6-tightest"
```

All other env vars identical to 046 verification. RESUME_FROM_CKPT same path.

## Acceptance

Reference = 046 verification quantized = 1.07467, **size 15,953,718**.

Per arm:
- **Strong legal win**: quantized < 1.0730 AND size ≤ 16,000,000 (would beat 046G-tightest legally)
- **Legal win**: quantized < 1.0739 AND size ≤ 16,000,000 (beats 046B-tight equivalent)
- **Neutral legal**: quantized 1.0739–1.0760 AND size ≤ 16,000,000
- **Illegal**: size > 16,000,000 (any quant value irrelevant)
- **Kill**: quantized > 1.0780 (low-bit emb damaged the model badly)

## What to watch

- **`Total submission size quantized+brotli:`** — MUST be ≤ 16,000,000 for an arm to count
- Quantized val_bpb
- Pure-emb arms tell us the raw cost of bit reduction
- Clip-retune arms tell us if tighter clip recovers some loss
- Stacked arms tell us if the SDClip wins survive low-bit emb

## Decision tree

| Outcome | Next |
|---|---|
| 046H-emb6-tightest wins legally (-0.0015+) | Lock as new baseline; consider EMBED_BITS=5 + tightest if size still has room |
| 046H-emb5-tightest wins legally even bigger | Push further: EMBED_BITS=4 sweep? Or final-stack with TTT |
| All emb6 arms regress | int6 emb too lossy on this stack; try LQER_RANK=8 on emb to compensate, or accept smaller win |
| Clip retune (clip12) helps significantly | Combine: low-bit emb + tight emb clip + tight matrix clip |

## Cost

~$11, ~90 min sequential on same NE-1 4×H100 pod.
