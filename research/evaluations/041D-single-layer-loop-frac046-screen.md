# Evaluation: 041D — single loop layer (layer 5 only), NUM_LOOPS=2, frac=0.46

**Date:** 2026-04-26
**Status:** KILL
**Spec:** `research/specs/041D-single-layer-loop-frac046-screen.md`
**Branch:** `exp/039b-loop-band-activation-screen` @ `5bbf12f`

## Result

| Metric | Value |
|--------|-------|
| Final step | 6057 |
| Loop activated | step 3011 (frac=0.460) |
| Recurrent steps | ~3046 |
| Pre-quant EMA val_bpb | **1.06993** |
| Peak VRAM | 32,995 MiB allocated / 38,268 MiB reserved |

Note: TRAINING_ONLY_SCREEN=1 — quantized val_bpb not available.

## Comparison

| Run | Steps | Loop start | Recurrent steps | Pre-quant EMA val_bpb | Δ vs baseline |
|-----|-------|-----------|-----------------|----------------------|---------------|
| Baseline (039b) | 5156 | step ~2300 (frac=0.35) | ~2867 | 1.06514 | — |
| 041A (4-5, NL=2, frac=0.46) | 5722 | step 3030 | ~2692 | 1.06545 | +0.00031 |
| 041Ab (4-5, NL=2, frac=0.63) | 5982 | step 4148 | ~1834 | 1.06736 | +0.00222 |
| 041B (3-5, NL=1, frac=0.59) | 6002 | step 3862 | ~2140 | 1.06842 | +0.00328 |
| 041Bb (3-5, NL=1, frac=0.65) | 6096 | step 4252 | ~1844 | 1.06836 | +0.00322 |
| **041D (5 only, NL=2, frac=0.46)** | **6057** | **step 3011** | **~3046** | **1.06993** | **+0.00479** |
| 040 (no loop) | 6569 | — | 0 | 1.07223 | +0.00709 |

## Training loss trajectory

| Step | 041D | Notes |
|------|------|-------|
| 1000 | ~2.76 | |
| 2000 | ~2.65 | |
| 3000 | 2.6166 | |
| 3011 | — | loop ON |
| 3100 | 2.4963 | activation (small spike) |
| 4000 | 2.4558 | |
| 5000 | 2.4786 | |
| 5700 | 2.3823 | |
| 6000 | 2.3596 | |
| 6057 | end | |

## Throughput

| Phase | tok/s |
|-------|-------|
| Pre-loop (phase 1) | ~4293K |
| Post-loop (phase 2) | ~4272K → ~3974K (cumulative-average artifact; true interval rate ~4100K) |

Phase 2 throughput was faster than the spec's 3727K prediction — the single-layer loop costs less than expected. This meant 041D did achieve ~3046 recurrent steps (+354 vs spec prediction of 3071, close).

## Analysis

The spec predicted val_bpb ~1.06462 based on a 2.2e-6 val_bpb/loop-step rate derived from 041A. Actual was 1.06993 — a miss of +0.00531. The prediction model failed because it assumed loop steps are equivalent regardless of layer count. They are not.

**The key finding:** 041D achieved ~3046 recurrent steps vs 041A's ~2692 — 354 more loop steps — yet landed at 1.06993 vs 041A's 1.06545. More loop steps with fewer loop layers is strictly worse. Looping only layer 5 means each extra pass recirculates through just 1 transformer layer; the recurrent representation has no depth to integrate. Two passes over a single layer adds less computation than two passes over two layers, both in FLOP terms and in representational capacity.

Comparing the single-layer and two-layer families at matched ~3000 recurrent steps:
- 041A (2 layers, ~2692 steps): 1.06545
- 041D (1 layer, ~3046 steps): 1.06993

The single-layer arm is 0.00448 worse despite more recurrent steps. **Recurrence depth (layers × passes) matters more than recurrent step count.**

The spec hypothesis — that throughput savings would buy enough extra loop steps to compensate for reduced width — was falsified. The quality of each loop step decays faster than the quantity improves.

## Verdict

**KILL** — 1.06993 is the worst result of the NUM_LOOPS=2 family, despite the most recurrent steps. This closes the single-layer loop path. The 041 family conclusion is clear: NUM_LOOPS=2 on layers 4-5 (041A) is the minimum viable recurrence configuration; reducing either the loop count (NL=1) or the layer count (1 layer) degrades quality faster than the throughput gain can recover.

The viable frontier remains 041A (1.06545, +0.00031 vs baseline) — essentially baseline-quality at late activation. Next direction: find a loop config that actively beats baseline, not just matches it.
