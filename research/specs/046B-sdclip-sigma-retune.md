# Spec 046B — SDClip sigma re-tune sweep

**Slug:** `046B-sdclip-sigma-retune`
**Created:** 2026-04-27
**Status:** READY (after 046 PASS)
**Branch:** `exp/046-quant-repair`
**Commit:** `0ea6a97`
**Parent:** spec 046 (verification), idea Q1 in `research/ideas/quant-repair.md`

## Hypothesis

Baseline already has per-category SDClip:
- `MATRIX_CLIP_SIGMAS=12.85` (matrix fallback)
- `ATTN_CLIP_SIGMAS=13.0` (attention)
- `MLP_CLIP_SIGMAS=12.0` (MLP)
- `EMBED_CLIP_SIGMAS=15.0` (embedding, int7)

These were tuned on the **pre-CaseOps** stack (PR #1586). CaseOps changed:
- Embedding shape (control tokens)
- Token frequency distribution (case ops add new pseudo-tokens)
- Likely shifted weight statistics in MLP/attn through training dynamics

Original optimum may have shifted. PR #1720's sensitivity analysis showed
loosening MLP-deep clip σ from 12 → 14 inverted post−pre quant gap from
−0.005 to **+0.011**. That's a ±0.016 swing on a single knob — confirms
the lever is sharp. Worth a sweep.

## Predicted outcome

Per literature + PR evidence: −0.001 to −0.003 BPB if the optimum has shifted
by ±0.5σ; could regress by similar magnitude if current values were near-optimal.

## Arms

Sweep around current per-category. Tighter and looser by ±0.5σ.

| Arm | MLP | ATTN | EMBED | Predicted |
|---|---|---|---|---|
| 046B-tight | 11.5 | 12.5 | 14.5 | tighter (smaller weights → fewer pruned, smaller post-quant) |
| 046B-loose | 12.5 | 13.5 | 15.5 | looser (more retention) |
| 046B-mlp-tight | 11.5 | 13.0 | 15.0 | only MLP tighter (most params) |
| 046B-attn-loose | 12.0 | 14.0 | 15.0 | only attn looser (small params, may benefit from looser) |

All arms run sequentially on same pod after 046 verification.

## Config diff (vs 046 verification baseline)

```bash
# arm 046B-tight
MATRIX_CLIP_SIGMAS=12.85
MLP_CLIP_SIGMAS=11.5
ATTN_CLIP_SIGMAS=12.5
EMBED_CLIP_SIGMAS=14.5
RUN_ID="046B-sdclip-tight"

# arm 046B-loose
MLP_CLIP_SIGMAS=12.5
ATTN_CLIP_SIGMAS=13.5
EMBED_CLIP_SIGMAS=15.5
RUN_ID="046B-sdclip-loose"

# arm 046B-mlp-tight
MLP_CLIP_SIGMAS=11.5
RUN_ID="046B-sdclip-mlp-tight"

# arm 046B-attn-loose
ATTN_CLIP_SIGMAS=14.0
RUN_ID="046B-sdclip-attn-loose"
```

All other env vars identical to 046 verification.

## Acceptance

Reference = 046 verification quantized (~1.07467).

Per arm:
- **Win**: < 1.0735
- **Noise**: 1.0735–1.0760
- **Kill**: > 1.0760

## What to watch

- Pruning percentage in GPTQ log (lower = tighter clip working)
- Submission size — tighter clip should compress slightly smaller
- Quantized val_bpb

## Decision

| Result | Next |
|---|---|
| One arm wins clearly | Adopt; combine with 046A winner if both win |
| 046B-tight wins, 046B-loose loses | Sweep further tight (11.0 / 12.0 / 14.0) |
| Opposite | Sweep further loose |
| All noise | SDClip is well-tuned even post-CaseOps; close direction |

## Cost

~$4 for 4 arms (~5-7 min each).
