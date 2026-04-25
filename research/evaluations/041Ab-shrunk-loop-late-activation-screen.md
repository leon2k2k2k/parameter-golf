# Evaluation: 041Ab — shrunk loop (layers 4-5), NUM_LOOPS=2, frac=0.63

**Date:** 2026-04-26
**Status:** KILL
**Spec:** `research/specs/041Ab-shrunk-loop-late-activation-screen.md`
**Branch:** `exp/039b-loop-band-activation-screen` @ `5bbf12f`

## Result

| Metric | Value |
|--------|-------|
| Final step | 5982 |
| Loop activated | step 4148 (frac=0.630) |
| Recurrent steps | ~1834 |
| Pre-quant EMA val_bpb | **1.06736** |
| Quantized val_bpb | **1.07663** |
| Submission size | 15,916,799 bytes |
| Peak VRAM | 37,445 MiB allocated / 42,708 MiB reserved |

## Comparison

| Run | Steps | Loop start | Recurrent steps | Pre-quant EMA val_bpb | Δ vs baseline |
|-----|-------|-----------|-----------------|----------------------|---------------|
| Baseline (039b) | 5156 | step ~2300 (frac=0.35) | ~2856 | 1.06514 | — |
| 041 (frac=0.46) | 5722 | step 3030 (frac=0.46) | ~2692 | 1.06545 | +0.00031 |
| **041Ab (frac=0.63)** | **5982** | **step 4148 (frac=0.63)** | **~1834** | **1.06736** | **+0.00222** |
| 040 (no loop) | 6569 | — | 0 | 1.07223 | +0.00709 |

## Training loss trajectory (selected steps)

| Step | Baseline | 041 | 041Ab |
|------|----------|-----|-------|
| 1000 | 2.7670 | 2.7652 | 2.7644 |
| 2000 | 2.6552 | 2.6510 | 2.6699 |
| 3000 | 2.5784 | 2.6151 | 2.6151 |
| 4000 | 2.4050 | 2.4387 | 2.4699 |
| 4148 | — | — | loop ON |
| 4800 | 2.2884 | 2.3469 | 2.3765 |
| 5100 | 2.3130 (end) | 2.3527 | 2.3848 |
| 5800 | — | 2.3279 | 2.3279 |
| 5900 | — | 2.3567 | 2.3567 |
| 5982 | — | — | end |

## Analysis

Later activation (frac=0.63 vs 0.46) gave 260 more steps (5982 vs 5722) but worse val_bpb (+0.00222 vs +0.00031). The longer no-loop phase meant only ~1834 recurrent steps vs 041's ~2692 — not enough recurrence to recover.

The training loss gap to baseline was already ~0.045 at loop activation (step 4148 at ~2.43 vs baseline's ~2.38 at step 4000). The short recurrent phase closed this to a tie at steps 5800–5900 in raw loss, but the EMA had already accumulated the deficit over the longer pre-loop phase.

**Key finding:** More pre-loop steps → deeper loss basin at activation, but fewer recurrent steps to exploit it. The tradeoff tips negative at frac=0.63. The sweet spot appears to be closer to frac=0.46 (spec 041), where the near-tie result suggests the basin/recurrence balance is roughly neutral.

## Verdict

**KILL** — val_bpb 1.06736 ≥ kill threshold 1.0670. Later activation with fewer recurrent steps is strictly worse than earlier activation (041). The shrunk-loop + late-activation family does not improve on baseline.
