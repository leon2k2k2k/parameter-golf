# Evaluation: 041F — remove layer 10 (NUM_LAYERS=10), loop layers 4-5, NUM_LOOPS=2, frac=0.46

**Date:** 2026-04-26
**Status:** KILL
**Spec:** `research/specs/041F-remove-last-layer-loop45-screen.md`
**Branch:** `exp/039b-loop-band-activation-screen` @ `5bbf12f`

## Result

| Metric | Value |
|--------|-------|
| Final step | 6137 |
| Loop activated | step 3286 (frac=0.46 ✅) |
| Recurrent steps | ~2851 |
| Pre-quant EMA val_bpb | **1.07203** |
| Peak VRAM | (not captured) |
| Model params | 33,058,905 (vs ~37M baseline — ~11% fewer) |

Note: TRAINING_ONLY_SCREEN=1 — quantized val_bpb not available.

## Comparison

| Run | Steps | Loop start | Recurrent steps | Pre-quant EMA val_bpb | Δ vs baseline |
|-----|-------|-----------|-----------------|----------------------|---------------|
| Baseline (039b) | 5156 | step ~2300 (frac=0.35) | ~2867 | 1.06514 | — |
| 041A (4-5, NL=2, frac=0.46) | 5722 | step 3030 | ~2692 | 1.06545 | +0.00031 |
| 041Ab (4-5, NL=2, frac=0.63) | 5982 | step 4148 | ~1834 | 1.06736 | +0.00222 |
| 041D (5 only, NL=2, frac=0.46) | 6057 | step 3011 | ~3046 | 1.06993 | +0.00479 |
| **041F (10L, 4-5, NL=2, frac=0.46)** | **6137** | **step 3286** | **~2851** | **1.07203** | **+0.00689** |
| 040 (no loop) | 6569 | — | 0 | 1.07223 | +0.00709 |

## Training loss trajectory

| Step | 041A | 041F | Δ |
|------|------|------|---|
| 1000 | ~2.77 | ~2.78 | +0.01 |
| 2000 | ~2.65 | ~2.68 | +0.023 |
| 3286 | — | — | loop ON |
| 3500 | 2.4330 | 2.4676 | +0.035 |
| 4000 | 2.4387 | 2.4759 | +0.037 |
| 4500 | 2.4323 | 2.4714 | +0.039 |
| 5000 | 2.4462 | 2.4902 | +0.044 |
| 5500 | 2.3940 | 2.4368 | +0.043 |

## Analysis

The spec hypothesis was that removing layer 10 would speed up throughput enough (~+185 loop steps vs 041A) to overcome the capacity loss. This failed conclusively.

**The throughput gain was real**: phase 1 ran at ~4688K tok/s (vs baseline's ~4292K) and phase 2 at ~4400K tok/s (vs 041A's ~3277K). 041F reached 6137 steps — more than 041A's 5722. Recurrent steps were ~2851, close to the predicted ~2877.

**The capacity loss was not neutral**: 041F ran ~+0.035–0.044 above 041A throughout the loop phase — a persistent, widening gap. The 10-layer model (33M params, ~11% fewer) simply cannot match the representational quality of the 11-layer model, regardless of loop steps.

The result (1.07203) nearly matches the no-loop ablation (040: 1.07223) — the loop benefit is nearly entirely cancelled by capacity loss. Removing a full transformer layer costs ~0.007 val_bpb, which far exceeds any gain from +159 extra loop steps at 2.2e-6/step (~0.00035).

## Verdict

**KILL** — 1.07203, worst result in the 2-pass family. The "remove a layer for speed" approach is closed: the capacity tax is ~20× larger than the loop-step benefit. 041G (shrink late MLP width instead of removing layers) is the cleaner alternative — keeps all 11 layers, trades a smaller fraction of capacity for a smaller throughput gain.
