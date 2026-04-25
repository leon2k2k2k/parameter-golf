# Evaluation 039bM — loop depth curriculum (1-loop → 2-loop at frac=0.65)

**Spec:** `research/specs/039bM-loop-depth-curriculum-screen.md`
**Status:** COMPLETE (wallclock cap, step 5431/20000)

## Result

| Metric | Value |
|--------|-------|
| pre-quant EMA val_bpb | **1.06646** |
| quantized val_bpb | **1.07596** |
| Δ pre-quant vs baseline (1.06514) | +0.00132 *(worse)* |
| Δ pre-quant vs 039bL (1.06594) | +0.00052 *(worse)* |
| Δ quant vs baseline (1.07410) | +0.00186 *(worse)* |

**Verdict: KILL** — worse than both baseline and 039bL. Depth curriculum did not help.

## Transition events

| Event | Step | frac | depth | Train loss | Δ vs baseline |
|-------|------|------|-------|-----------|--------------|
| Loop activation (1-loop) | 2296 | 0.350 | 2 | 2.7096 | −0.010 *(no spike)* |
| Depth upgrade (2-loop) | 3882 | 0.650 | 3 | — | — |
| Step 3900 (post-upgrade) | 3900 | — | 3 | 2.5428 | +0.005 *(tiny spike)* |

## Step-matched train loss

| Step | Baseline | 039bL | 039bM | Δ bM vs baseline | Notes |
|------|----------|-------|-------|-----------------|-------|
| 1400 | 2.5973 | 2.5589 | 2.5608 | −0.037 | |
| 2000 | 2.6552 | 2.5894 | 2.5926 | −0.063 | |
| 2300 | 2.7198 | 2.9334 | 2.7096 | −0.010 | **1-loop: no spike** |
| 2500 | 2.5104 | 2.4366 | 2.4449 | −0.066 | |
| 3000 | 2.5784 | 2.5182 | 2.5330 | −0.045 | |
| **3882** | — | — | — | — | **depth upgrade: 1→2 loops** |
| 3900 | 2.5377 | 2.5084 | 2.5428 | +0.005 | small post-upgrade spike |
| 4000 | 2.4050 | 2.3776 | 2.3941 | −0.011 | recovered |
| 4500 | 2.3854 | 2.3844 | 2.4000 | +0.015 | |
| 5100 | 2.3130 | 2.3294 | 2.3449 | +0.032 | eroding |
| 5300 | — | — | 2.4818 | — | (bM ran 272 extra steps) |

## Analysis

The depth curriculum achieved its mechanical goal: both transitions were smooth.
The 1-loop activation at step 2296 produced essentially no spike (2.7096 vs baseline 2.7198,
Δ = −0.010) — far better than 039bL's 2-loop activation (spike to 2.9334). The depth upgrade
at step 3882 caused only a tiny perturbation (+0.005 at step 3900, gone by step 4000).

Despite the cleaner transitions, 039bM tracked 039bL almost exactly through the mid-run
(steps 2500–3800), and finished worse. The pre-loop advantage (−0.063 at step 2000) eroded
on the same trajectory as 039bL, with the deficit turning positive by step 4500. The
floor_then_linear second-half LR starvation problem dominates regardless of loop activation
strategy.

The depth curriculum ran ~1500 steps at 1-loop throughput (~4,250K tok/s vs 2-loop
~3,870K), so the run was slightly cheaper per step. This didn't translate to better
convergence — the model had the same LR budget regardless of depth schedule.

## Conclusion

039bM = 039bL + smoother loop activation. The smoother activation is real (+0.00052 BPB
worse than 039bL, which is within noise for the mechanism difference), but neither beats
baseline. The depth curriculum does not rescue floor_then_linear. The binding constraint
remains the same: linear second-half LR (0.400→0.100) starves late convergence.

## Series summary (039bG through 039bM)

| Arm | Schedule | Carry | Loop start | pre-quant BPB | Δ vs baseline |
|-----|----------|-------|-----------|--------------|---------------|
| baseline | WSD | enabled | frac=0.35 (2-loop) | 1.06514 | — |
| 039bG | first_half_floor | enabled | frac=0.35 (2-loop) | 1.06800 | +0.00286 |
| 039bI | floor_then_wsd | enabled | frac=0.35 (2-loop) | 1.06622 | +0.00108 |
| 039bJ | floor_then_linear | enabled | frac=0.35 (2-loop) | 1.06598 | +0.00084 |
| 039bK | WSD | disabled | frac=0.35 (2-loop) | killed (neutral) | ~0 |
| 039bL | floor_then_linear | disabled | frac=0.35 (2-loop) | 1.06594 | +0.00080 |
| 039bM | floor_then_linear | disabled | frac=0.35 (1-loop) → frac=0.65 (2-loop) | 1.06646 | +0.00132 |

No LR schedule in this series beats baseline. Standard WSD remains optimal.
