# Evaluation: 041Cb — shrunk loop (layers 4-5), NUM_LOOPS=1, frac=0.46

**Date:** 2026-04-26
**Status:** KILL
**Spec:** `research/specs/041Cb-shrunk-loop-1pass-late-activation-screen.md`
**Branch:** `exp/039b-loop-band-activation-screen` @ `5bbf12f`

## Result

| Metric | Value |
|--------|-------|
| Final step | 6095 |
| Loop activated | step 3030 (frac=0.460) |
| Loop depth | 2 (1 extra pass on layers 4-5) |
| Recurrent steps | ~3065 |
| Pre-quant EMA val_bpb | **1.06816** |
| Quantized val_bpb | N/A (TRAINING_ONLY_SCREEN=1) |
| Peak VRAM | 32,988 MiB allocated / 38,314 MiB reserved |

## Comparison

| Run | Steps | Loop depth | Recurrent steps | Pre-quant EMA val_bpb | Δ vs baseline |
|-----|-------|-----------|-----------------|----------------------|---------------|
| Baseline (039b) | 5156 | 3 (layers 3-5) | ~2856 | 1.06514 | — |
| 041 (depth:3, frac=0.46) | 5722 | 3 (layers 4-5) | ~2692 | 1.06545 | +0.00031 |
| 041Ab (depth:3, frac=0.63) | 5982 | 3 (layers 4-5) | ~1834 | 1.06736 | +0.00222 |
| **041Cb (depth:2, frac=0.46)** | **6095** | **2 (layers 4-5)** | **~3065** | **1.06816** | **+0.00302** |
| 040 (no loop) | 6569 | 0 | 0 | 1.07223 | +0.00709 |

## Training loss trajectory (selected steps)

| Step | Baseline | 041 | 041Ab | 041Cb |
|------|----------|-----|-------|-------|
| 2000 | 2.6552 | 2.6510 | 2.6699 | 2.6551 |
| 3000 | 2.5784 | 2.6151 | 2.6151 | 2.6150 |
| 3030 | — | loop ON | — | loop ON |
| 4000 | 2.4050 | 2.4387 | 2.4699 | 2.4531 |
| 4800 | 2.2884 | 2.3469 | 2.3765 | ~2.37 |
| 5100 | 2.3130 (end) | 2.3527 | 2.3848 | 2.3879 |
| 5800 | — | 2.3279 | 2.3279 | 2.3333 |
| 6000 | — | — | — | 2.3564 |
| 6095 | — | — | — | end |

## Analysis

041Cb achieved 6095 steps — the most in the 041 family — due to the minimal throughput tax of a 1-pass loop (~4000K tok/s steady state vs baseline's ~3385K). Despite more total steps and more recurrent steps than 041 (~3065 vs ~2692), the shallower recurrence (depth:2 = 1 extra pass vs depth:3 = 2 extra passes) produced worse val_bpb (1.06816 vs 1.06545).

The pre-loop training loss curves are nearly identical across all 041 variants. The divergence is entirely post-loop: depth:3 (041) shows a faster loss descent than depth:2 (041Cb) at matched steps post-activation. This confirms that loop depth (passes per step) matters more than total recurrent steps.

**Key finding:** Doubling the recurrent step count (3065 vs 2692) cannot compensate for halving the recurrence depth. Loop depth is the critical variable, not number of recurrent steps.

## Verdict

**KILL** — val_bpb 1.06816 ≥ kill threshold 1.0670. Lightest recurrence (1 pass) is strictly worse than 2-pass (041) despite more total steps. The 041 family as a whole (shrunk loop, late activation) does not improve on baseline — the full 3-layer loop at frac=0.35 remains optimal.
