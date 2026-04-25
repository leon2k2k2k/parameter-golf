# Evaluation 039bJ — floor_then_linear LR schedule

**Spec:** `research/specs/039bJ-floor-then-linear-screen.md`
**Status:** COMPLETE (wallclock cap, step 5155/20000)

## Result

| Metric | Value |
|--------|-------|
| pre-quant EMA val_bpb | **1.06598** |
| quantized val_bpb | **1.07528** |
| Δ pre-quant vs baseline (1.06514) | +0.00084 *(worse)* |
| Δ quant vs baseline (1.07410) | +0.00118 *(worse)* |

**Verdict: KILL** — worse than baseline on both metrics. floor_then_linear does not help.

## Step-matched train loss

| Step | Baseline | 039bG | 039bJ | Δ bJ vs baseline |
|------|----------|-------|-------|-----------------|
| 1400 | 2.5973 | 2.5637 | 2.5619 | −0.035 |
| 1700 | 2.6351 | 2.5904 | 2.5888 | −0.046 |
| 2300 | 2.7198 | 2.8194 | 3.0009 | **+0.281** *(spike)* |
| 2600 | 2.5040 | 2.4039 | 2.4373 | −0.067 |
| 3000 | 2.5784 | 2.4515 | 2.5167 | −0.062 |
| 4000 | 2.4050 | 2.3323 | 2.3785 | −0.027 |
| 4700 | 2.3698 | 2.3680 | 2.3832 | +0.013 |
| 5100 | 2.3130 | 2.3338 | 2.3270 | +0.014 |

## Why it failed

Floor_then_linear removed the LR *jump* from 039bI (no discontinuity at step 2299), but the LR at loop activation was still 0.400 — well above MIN_LR. This caused a smaller but still damaging spike (3.0009 vs 039bI's 3.0641).

After the spike, 039bJ showed a real mid-run advantage (−0.062 to −0.067 vs baseline at steps 2600–3000), driven by the fast first-half LR descent giving better early representations. However, the lower second-half LR (0.400→0.100 linear vs baseline's 0.867→0.100 WSD) starved late convergence. The advantage eroded steadily, reaching parity (~0) by step 4500 and slightly behind baseline by step 5000.

## Key insight

The floor_then_linear shape front-loads learning (good) but the linear second-half is too conservative — the model doesn't have enough LR headroom in the final 13m to fully exploit what it learned in the first 7m. The net effect is a wash: better mid-run, worse late-run, neutral final BPB.

## Conclusion

floor_then_linear is structurally neutral-to-negative. The fundamental problem is that any LR > MIN_LR at loop activation causes a spike, and floor_then_linear at LR_REWARM_AT=0.35 has LR=0.400 at that point. The only schedule that avoids the spike entirely is first_half_floor (039bG), which has LR=MIN_LR at loop activation. However, first_half_floor has its own problem: it starves the second half even more aggressively, ending at 1.06800 (worse than baseline 1.06514 despite better train loss trajectory).
