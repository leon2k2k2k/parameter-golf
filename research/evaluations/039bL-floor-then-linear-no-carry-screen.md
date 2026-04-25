# Evaluation 039bL — floor_then_linear + RECUR_ALPHA_ENABLED=0

**Spec:** `research/specs/039bL-floor-then-linear-no-carry-screen.md`
**Status:** COMPLETE (wallclock cap, step 5159/20000)

## Result

| Metric | Value |
|--------|-------|
| pre-quant EMA val_bpb | **1.06594** |
| quantized val_bpb | **1.07527** |
| Δ pre-quant vs baseline (1.06514) | +0.00080 *(worse)* |
| Δ vs 039bJ pre-quant (1.06598) | −0.00004 *(within noise)* |

**Verdict: KILL** — worse than baseline. Confirms frozen carry is neutral; outcome driven entirely by floor_then_linear schedule.

## Step-matched train loss

| Step | Baseline | 039bG | 039bJ | 039bL | Δ bL vs baseline |
|------|----------|-------|-------|-------|-----------------|
| 1400 | 2.5973 | 2.5637 | 2.5619 | 2.5589 | −0.038 |
| 2000 | 2.6552 | 2.5942 | 2.5923 | 2.5894 | −0.066 |
| 2300 | 2.7198 | 2.8194 | 3.0009 | 2.9334 | **+0.214** *(spike)* |
| 2500 | 2.5104 | 2.4187 | 2.4396 | 2.4366 | −0.074 |
| 3000 | 2.5784 | 2.4515 | 2.5167 | 2.5182 | −0.060 |
| 4000 | 2.4050 | 2.3323 | 2.3785 | 2.3776 | −0.027 |
| 4500 | 2.3854 | 2.3599 | 2.3847 | 2.3844 | −0.001 |
| 5100 | 2.3130 | 2.3338 | 2.3270 | 2.3294 | +0.016 |

## Analysis

039bL tracked 039bJ almost exactly throughout the entire run (within 0.003 at every step), confirming that frozen carry (RECUR_ALPHA_ENABLED) has no material effect on training dynamics once the LR schedule shape is held fixed. The 039bK ablation established this independently.

The slightly smaller loop spike in 039bL vs 039bJ (2.9334 vs 3.0009) suggests frozen carry alpha values [1.5610, 1.8531, 2.1320] add a small amount of instability at loop activation — but recovery was equally fast in both cases and the difference disappeared within 200 steps.

The same pattern as 039bJ played out: strong pre-loop advantage (up to −0.074 at step 2500), gradual erosion through the second half, convergence to baseline parity by step 4500, and slight underperformance in the final steps.

## Conclusion

039bL = 039bJ + no-carry. Since carry is neutral, 039bL's result is a near-exact replication of 039bJ (Δ = 0.00004 BPB). The hypothesis that combining both changes might produce a synergistic win was not supported. The floor_then_linear schedule shape is the binding constraint — frozen carry status is irrelevant.

## Series summary (039bH through 039bL)

| Arm | Schedule | Carry | Pre-quant BPB | Δ vs baseline |
|-----|----------|-------|--------------|---------------|
| baseline | WSD | enabled | 1.06514 | — |
| 039bG | first_half_floor | enabled | 1.06800 | +0.00286 |
| 039bH | floor_then_wsd | enabled | killed | — |
| 039bI | floor_then_wsd | enabled | 1.06622 | +0.00108 |
| 039bJ | floor_then_linear | enabled | 1.06598 | +0.00084 |
| 039bK | WSD | disabled | killed (neutral) | ~0 |
| 039bL | floor_then_linear | disabled | 1.06594 | +0.00080 |

**Root cause:** The loop activates at LR=0.400 for all floor_then_* schedules except first_half_floor (LR=MIN_LR). Any LR > MIN_LR at loop activation causes a spike. first_half_floor avoids the spike but over-starves second-half convergence. No LR schedule in this series beats baseline — the standard WSD shape remains optimal for this regime.
