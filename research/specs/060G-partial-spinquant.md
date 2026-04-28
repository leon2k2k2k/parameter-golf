# Spec 060G — Partial SpinQuant from PR #1898 on 060A baseline

**Status: DEPRECATED 2026-04-29 — empirically refuted by PR #1898 itself.**

PR #1898 ran this exact lever (Partial SpinQuant + EMBED_BITS=6 reinvest) on
its own base (#1851 at 1.06128) and got **1.06614** — a **regression of
+0.00486 BPB**. Their framing of "−0.01486 vs merged SOTA #1493 (1.0810)"
is misleading; the like-for-like comparison vs their actual parent shows
the lever doesn't help.

Reason to keep this spec on file: documentation of why we're NOT pursuing
SpinQuant on the 060A line. If the lever ever becomes promising in the
future (e.g., paired with deploy-time repair or different bit allocations),
the spec is here as a starting point. Do not run.

---

**Date:** 2026-04-29
**Branch:** `exp/060G-partial-spinquant` (forked from research)
**Parent:** 060A + #1898 SpinQuant code (port from `X-Abhishek-X/parameter-golf` PR #1898).

## Hypothesis

PR #1898 (X-Abhishek-X) reports val_bpb 1.06614 on #1851 base by **rotating only attention layers 5..N-1** (skipping early/late layers). Saves ~800 KB brotli compression vs full SpinQuant (which is broken in our 046C anyway). With saved bytes, can bump quantization aggressiveness. On 060A base (which is stronger than #1851), partial SpinQuant should produce similar or larger savings → reinvest in `EMBED_BITS=6` or LQER rank.

## Baseline

060A.

## Expected Δ

- Pure rotation (no reinvest): **~−0.0 BPB** (Hadamard rotation is loss-preserving)
- With reinvest into `EMBED_BITS=6` + LQER `RANK=5`: **−0.005 to −0.015 BPB**

## Accept criteria

- val_bpb ≤ (060A − 0.003)
- artifact ≤ 16 MB
- no quantization quality regression in attn layers

## Config diff vs 060A

```
SPINQUANT_ENABLED=1
SPINQUANT_START_LAYER=5
SPINQUANT_END_LAYER=10  # for 11-layer model (NUM_LAYERS=11; rotates layers 5..10)

# Reinvest saved bytes:
EMBED_BITS=6  (was 7)        — costs ~−40 KB on artifact (smaller embed) but needs more LQER
LQER_RANK=5   (was 4)        — costs ~+50 KB
LQER_TOP_K=4  (was 3)        — costs ~+40 KB
```

## Code changes

Real code port. Cherry-pick from PR #1898's `train_gpt.py`:

1. SpinQuant Hadamard rotation function (~50 lines)
2. Layer-range selector (rotate only layers 5..N-1) (~20 lines)
3. Hook in serialize/quantize path (~20 lines)
4. Hyperparameters fields (~5 lines)

Total ~100 lines. Estimate: 60-90 min careful port + smoke test.

## Hardware ladder

- 4×H100, **needs full re-train** (rotation affects training-time weight space).
  - OR can RESUME_FROM_CKPT if rotation is purely post-training (need to verify with #1898's code).
- Worst case: ~$8 (full retrain), best case: ~$3 (eval-only).

## Seed plan

1 seed (42).

## Stop-early criteria

- Compile fail on rotated path → kill, skip this spec
- val_bpb > 060A → kill (rotation regressed)

## Open questions

1. **Rotation: training-time or post-training only?** Determines whether we need full retrain ($8) or eval-only ($3).
2. **Does the EMBED_BITS=6 fit**? 6-bit embed table: ~7 MB raw vs ~9 MB at 7-bit. Need to verify our LQER patches recover the precision.
3. **Numerical stability under rotation + LQER**: untested combination. Smoke first.

## Phase 2 (NOT in 060G)

- 060G+: stack 060G with 060B (SDClip) and/or 060C (deploy-time repair).
