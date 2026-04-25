# Evaluation: 041Bb — original loop (layers 3-5), NUM_LOOPS=1, frac=0.65

**Date:** 2026-04-26
**Status:** KILL
**Spec:** (no spec file — run was launched with frac=0.65 as a corrected rerun of 041B)
**Branch:** `exp/039b-loop-band-activation-screen` @ `5bbf12f`

## Result

| Metric | Value |
|--------|-------|
| Final step | 6096 |
| Loop activated | step 4252 (frac=0.650) |
| Recurrent steps | ~1844 |
| Pre-quant EMA val_bpb | **1.06836** |
| Peak VRAM | 35,271 MiB allocated / 40,530 MiB reserved |

Note: TRAINING_ONLY_SCREEN=1 — quantized val_bpb not available.

## Comparison

| Run | Steps | Loop start | Recurrent steps | Pre-quant EMA val_bpb | Δ vs baseline |
|-----|-------|-----------|-----------------|----------------------|---------------|
| Baseline (039b) | 5156 | step ~2300 (frac=0.35) | ~2867 | 1.06514 | — |
| 041A (4-5, NL=2, frac=0.46) | 5722 | step 3030 | ~2692 | 1.06545 | +0.00031 |
| 041B (3-5, NL=1, frac=0.59) | 6002 | step 3862 | ~2140 | 1.06842 | +0.00328 |
| **041Bb (3-5, NL=1, frac=0.65)** | **6096** | **step 4252** | **~1844** | **1.06836** | **+0.00322** |
| 041Ab (4-5, NL=2, frac=0.63) | 5982 | step 4148 | ~1834 | 1.06736 | +0.00222 |
| 040 (no loop) | 6569 | — | 0 | 1.07223 | +0.00709 |

## Training loss trajectory

| Step | 041Bb | Notes |
|------|-------|-------|
| 1000 | 2.7616 | |
| 2000 | 2.6548 | |
| 3000 | 2.6141 | |
| 4000 | 2.4718 | |
| 4252 | — | loop ON |
| 4300 | 2.5777 | activation spike |
| 5000 | 2.4811 | |
| 6000 | 2.3522 | |
| 6096 | end | |

## Analysis

041Bb is the corrected-frac rerun of 041B (frac=0.65 to target 6096 steps vs 041B's 6002). The result is nearly identical: 1.06836 vs 1.06842 — confirming the result is not a frac artifact but a depth artifact.

**The finding is clean:** NUM_LOOPS=1 on layers 3-5 is strictly insufficient. Even with ~6100 steps — 940 more than baseline — val_bpb is 0.00322 worse. The extra steps don't compensate for shallow recurrence.

Comparing 041Bb vs 041Ab (same ~1844 recurrent steps, same ~6000 total steps, same late activation):
- 041Bb: NUM_LOOPS=1, 3 layers → val_bpb 1.06836
- 041Ab: NUM_LOOPS=2, 2 layers → val_bpb 1.06736

The NUM_LOOPS=2 arm is better by 0.00100 with identical recurrent step counts. **Depth (extra passes per loop) matters more than coverage (number of layers looped).** One extra pass is simply not enough recurrence to move the needle regardless of layer coverage.

## Verdict

**KILL** — val_bpb 1.06836 well above any useful threshold. This closes the NUM_LOOPS=1 family: both 3-layer and 2-layer variants at ~6100 steps land around 1.067-1.068. The only viable path is NUM_LOOPS=2, and the real lever is reducing N to get more loop steps per wallclock second (→ 041D).
