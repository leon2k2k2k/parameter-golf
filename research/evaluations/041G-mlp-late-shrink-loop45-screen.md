# Evaluation: 041G — shrink late MLP (layers 6-10, mult=3.0), loop layers 4-5, NUM_LOOPS=2, frac=0.46

**Date:** 2026-04-26
**Status:** KILL
**Spec:** `research/specs/041G-mlp-late-shrink-loop45-screen.md`
**Branch:** `exp/039b-loop-band-activation-screen` @ `92886ff`

## Result

| Metric | Value |
|--------|-------|
| Final step | 5793 |
| Loop activated | step 3102 (frac=0.46 ✅) |
| Recurrent steps | ~2691 |
| Pre-quant EMA val_bpb | **1.07180** |
| Peak VRAM | 36,477 MiB allocated / 41,718 MiB reserved |
| Model params | 35,944,647 (vs ~37M baseline — ~3% fewer) |

Note: TRAINING_ONLY_SCREEN=1 — quantized val_bpb not available.

## Comparison

| Run | Steps | Loop start | Recurrent steps | Pre-quant EMA val_bpb | Δ vs baseline |
|-----|-------|-----------|-----------------|----------------------|---------------|
| Baseline (039b) | 5156 | step ~2300 (frac=0.35) | ~2867 | 1.06514 | — |
| 041A (4-5, NL=2, frac=0.46) | 5722 | step 3030 | ~2692 | 1.06545 | +0.00031 |
| 041Ab (4-5, NL=2, frac=0.63) | 5982 | step 4148 | ~1834 | 1.06736 | +0.00222 |
| 041D (5 only, NL=2, frac=0.46) | 6057 | step 3011 | ~3046 | 1.06993 | +0.00479 |
| 041F (10L, 4-5, NL=2, frac=0.46) | 6137 | step 3286 | ~2851 | 1.07203 | +0.00689 |
| **041G (11L shrunk, 4-5, NL=2, frac=0.46)** | **5793** | **step 3102** | **~2691** | **1.07180** | **+0.00666** |
| 040 (no loop) | 6569 | — | 0 | 1.07223 | +0.00709 |

## Training loss trajectory

| Step | 041A | 041G | Δ |
|------|------|------|---|
| 500  | 2.6606 | 2.6630 | +0.002 |
| 1000 | ~2.77 | ~2.77 | ~0 |
| 2000 | ~2.65 | ~2.67 | +0.012 |
| 3102 | — | — | loop ON |
| 3500 | 2.4330 | 2.4480 | +0.015 |
| 4000 | 2.4387 | 2.4565 | +0.018 |
| 4500 | 2.4323 | 2.4476 | +0.015 |
| 5000 | 2.4462 | 2.4639 | +0.018 |
| 5500 | 2.3940 | 2.4089 | +0.015 |

## Analysis

041G was the softer capacity-reduction approach vs 041F — shrinking late MLP width (layers 6-10, mult=4→3) rather than removing a full layer. The training loss gap to 041A was only ~+0.015–0.018 throughout, far smaller than 041F's +0.034–0.044. The late MLP reduction appeared nearly neutral in train loss.

**Yet the final val_bpb (1.07180) is nearly identical to 041F (1.07203)** — and both are ~+0.007 above baseline. The throughput gain was real (phase 2 at ~4300K tok/s vs 041A's ~3277K), but recurrent steps were ~2691 — essentially the same as 041A's 2692. The wider model ran fewer total steps (5793 vs 5722) due to the faster no-loop phase eating less wallclock fraction.

The puzzle: train loss is ~0.015 above 041A but val_bpb is +0.007 worse than baseline. Two explanations:
1. **Generalization gap**: smaller late MLPs reduce the model's generalization capacity, visible in val but not train
2. **Quant sensitivity**: late MLP weights at mult=3 may be harder to quantize accurately, widening the pre-quant EMA gap beyond what train loss predicts

Either way, the result is unambiguous: shrinking late MLPs to gain loop throughput does not work. The capacity cost in val_bpb (+0.007) exceeds any benefit from comparable loop step counts.

## Verdict

**KILL** — 1.07180, nearly identical to 041F (1.07203) and far above baseline. Both "reduce non-loop compute" approaches (remove layer / shrink late MLPs) produce the same ~+0.007 val_bpb penalty regardless of mechanism. The 041 arc conclusion is now firm: **you cannot buy loop steps by shrinking non-loop capacity — the val_bpb cost always exceeds the gain.** The only viable path forward is finding configurations that beat baseline without trading capacity.
