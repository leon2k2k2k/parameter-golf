# Spec 046D — LQER knobs sweep (TOP_K + ASYM_GROUP)

**Slug:** `046D-lqer-knobs-sweep`
**Created:** 2026-04-27
**Status:** READY (after 046 PASS)
**Branch:** `exp/046-quant-repair`
**Commit:** `0ea6a97`
**Parent:** spec 046 (verification), ideas Q5b/Q5c in `research/ideas/quant-repair.md`

## Hypothesis

LQER on the record track is **a single-config technique** (PR #1797 only):
`RANK=4 TOP_K=3 ASYM_ENABLED=1 ASYM_GROUP=64`. Zero hyperparam sweeps exist.

Two cheap knobs to probe:

1. **TOP_K=3 → 5 or 8**: currently we apply LQER residual to only the 3 highest-residual
   tensors (likely tok_emb + 2 MLP fc). Coverage to more tensors costs ~10KB per added
   tensor (4 ranks × dim × bytes), well within our ~46KB headroom.
2. **ASYM_GROUP=64 → 32 or 128**: the group size for B-factor scales. Finer (32)
   improves reconstruction at modest byte cost; coarser (128) saves bytes for
   coverage growth.

If the LQER curve is slope-favorable in either direction, this is free signal.

## Predicted outcome

- TOP_K sweep: −0.001 to −0.003 if coverage matters; null if top-3 already
  captures most residual mass
- ASYM_GROUP sweep: −0.0005 to −0.0015 if 64 was a one-shot guess; may regress
  if 128 saves bytes that don't matter

## Arms

| Arm | LQER_TOP_K | LQER_ASYM_GROUP | Cost (bytes) | Predicted |
|---|---|---|---|---|
| 046D-topk5 | 5 | 64 | ~+15-20KB | -0.001 to -0.002 |
| 046D-topk8 | 8 | 64 | ~+30-40KB (cap-tight) | -0.001 to -0.003 |
| 046D-group32 | 3 | 32 | ~+5KB | -0.0005 to -0.0015 |
| 046D-group128 | 3 | 128 | ~-2KB | ±0.001 |
| 046D-rank6 | 3 (rank=6) | 64 | ~+15-20KB | -0.001 to -0.002 |

(Add LQER_RANK=6 as bonus — also currently untouched.)

## Config diff (vs 046 verification baseline)

```bash
# 046D-topk5
LQER_TOP_K=5
RUN_ID="046D-lqer-topk5"

# 046D-topk8 — risk: may exceed 16MB cap
LQER_TOP_K=8
RUN_ID="046D-lqer-topk8"

# 046D-group32
LQER_ASYM_GROUP=32
RUN_ID="046D-lqer-group32"

# 046D-group128
LQER_ASYM_GROUP=128
RUN_ID="046D-lqer-group128"

# 046D-rank6
LQER_RANK=6
RUN_ID="046D-lqer-rank6"
```

All other env vars identical to 046 verification.

## Acceptance

Reference = 046 verification quantized (~1.07467).

Per arm:
- **Win**: < 1.0735
- **Strong win**: < 1.0730
- **Noise**: 1.0735–1.0760
- **Kill**: > 1.0760
- **Cap overflow**: submission size > 16,777,216 bytes → arm invalid (LQER bytes exceeded headroom)

## What to watch

- `LQER:` log lines at startup confirming new TOP_K/ASYM_GROUP values
- `Total submission size quantized+brotli:` — must stay under 16MB
- Quantized val_bpb (the headline)
- For TOP_K=8: which 8 tensors got LQER (log will show)

## Decision

| Result | Next |
|---|---|
| TOP_K sweep wins → adopt, defer rank/group |
| ASYM_GROUP=32 wins → also try 16 |
| Cap overflows on TOP_K=8 → manage budget by reducing RANK or moving emb back to int6 |
| All noise → LQER curve is flat, close direction |

## Cost

~$5 for 5 arms (~5-7 min each).
