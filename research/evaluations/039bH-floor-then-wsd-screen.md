# Evaluation 039bH — floor_then_wsd + loop@0.175

**Spec:** `research/specs/039bH-floor-then-wsd-screen.md`
**Status:** KILLED early (12.9m of 20m wallclock, step 3200)
**Killed by:** user — misalignment signal

## Result

Run killed at step 3200 / 12.9m. No final val_bpb produced.

**Last observed train loss vs baseline at matched steps:**

| Step | baseline | 039bG | 039bH | 039bE |
|---|---|---|---|---|
| 1300 | 2.7537 | 2.7246 | 2.7043 (Δ −0.049) | — |
| 2000 | 2.6552 | 2.5942 | 2.6441 (Δ −0.011) | — |
| 2600 | 2.5040 | 2.4039 | 2.4810 (Δ −0.023) | 2.3526 |
| 3100 | 2.4598 | 2.3429 | 2.4353 (Δ −0.025) | 2.3357 |

(039bH wallclock was ~1.7m behind baseline/039bG at matched steps due to loop@0.175 throughput cost.)

## Why killed

039bH showed a small consistent advantage over baseline (~−0.02 to −0.05 train loss) but was clearly underperforming both:
- **039bG** (Δ −0.12 at step 3100) — same LR schedule but loop@0.35
- **039bE** (Δ −0.12 at step 3100) — same loop@0.175 but first_half_floor

The early loop activation in 039bH combined with floor_then_wsd was not producing any synergy. The 039bH curve was running *between* baseline and 039bG at matched steps — suggesting the LR floor phase and early loop are partially canceling each other, not compounding. The WSD resumption at 35% wallclock (~7m) did not visibly accelerate convergence vs the plateau seen in 039bE.

## Conclusion

**039bH is misaligned.** Two simultaneous changes (early loop + new LR schedule) make attribution impossible, and neither combination is beating the simpler 039bG result. The hypothesis that early loop + floor_then_wsd synergize was not supported.

## Decision: Kill → 039bI

039bI runs floor_then_wsd with loop@0.35 (no early loop cost). This is the clean single-change isolation:
- If 039bI beats baseline: floor_then_wsd is the driver; early loop is irrelevant/harmful
- If 039bI also loses: floor_then_wsd doesn't help; kill the schedule

039bI launched immediately after kill on the same pod.
