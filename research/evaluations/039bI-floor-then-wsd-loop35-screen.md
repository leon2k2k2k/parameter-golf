# Evaluation 039bI — floor_then_wsd + loop@0.35

**Spec:** `research/specs/039bI-floor-then-wsd-loop35-screen.md`
**Status:** COMPLETE (wallclock cap, step 5155/20000)

## Result

| Metric | Value |
|--------|-------|
| pre-quant EMA val_bpb | **1.06622** |
| quantized val_bpb | **1.07511** |
| Δ pre-quant vs baseline (1.06514) | +0.00108 *(worse)* |
| Δ quant vs baseline (1.07410) | +0.00101 *(worse)* |

**Verdict: KILL** — worse than baseline on both metrics. floor_then_wsd does not help.

## Step-matched train loss

| Step | Baseline | 039bG | 039bI | Δ bI vs baseline |
|------|----------|-------|-------|-----------------|
| 400 | 2.6514 | 2.6461 | 2.6569 | +0.006 |
| 1300 | 2.7537 | 2.7246 | 2.7580 | +0.004 |
| 2300 | 2.7198 | 2.8194 | 3.0641 | **+0.344** *(spike)* |
| 2700 | 2.6610 | 2.5393 | 2.6633 | +0.002 |
| 3100 | 2.4598 | 2.3429 | 2.4625 | +0.003 |

## Why it failed

039bI had two destabilizing events simultaneously at step ~2299 (~7m):
1. **Loop activation** (ENABLE_LOOPING_AT=0.35)
2. **LR rewarm jump** (LR jumps from ~0.400 back up to ~0.867 at LR_REWARM_AT=0.35)

The combined shock caused a massive spike (train loss 3.0641 vs baseline 2.7198, +0.344). After recovery, 039bI converged to near-baseline parity but could never make up the lost ground, ending +0.00108 BPB worse.

## Conclusion

The floor_then_wsd schedule is incompatible with simultaneous loop activation. The LR discontinuity at the switch point is the root cause — any large upward LR step at the same time as loop activation causes a compounding shock. Led to 039bJ (smooth linear continuation, no LR jump).
