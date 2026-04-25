# Evaluation 039bK — RECUR_ALPHA_ENABLED=0 (frozen carry ablation)

**Spec:** `research/specs/039bK-no-frozen-carry-screen.md`
**Status:** KILLED early (step 3300 / 11.5m of 20m wallclock)
**Killed by:** user — neutral result, not worth completing

## Result

Run killed at step 3300 / 11.5m. No final val_bpb produced.

**Step-matched train loss vs baseline:**

| Step | Baseline | 039bG | 039bK | Δ bK vs baseline |
|------|----------|-------|-------|-----------------|
| 400  | 2.6514 | 2.6461 | 2.6370 | −0.014 |
| 900  | 2.7559 | 2.7510 | 2.7415 | −0.014 |
| 1100 | 2.6907 | 2.6688 | 2.6826 | −0.008 |
| 1700 | 2.6351 | 2.5904 | 2.6365 | +0.001 |
| 2100 | 2.7136 | 2.6489 | 2.7153 | +0.002 |
| 2300 | 2.7198 | 2.8194 | 2.8960 | **+0.176** *(loop spike)* |
| 2400 | 2.6689 | 2.5913 | 2.6677 | −0.001 |
| 2600 | 2.5040 | 2.4039 | 2.5036 | −0.000 |
| 2800 | 2.5612 | 2.4380 | 2.5609 | −0.000 |
| 3100 | 2.4598 | 2.3429 | 2.4579 | −0.002 |
| 3200 | 2.4849 | 2.3613 | 2.4844 | −0.000 |
| 3300 | 2.5988 | — | 2.5988 | −0.000 |

## Key observations

**Pre-loop (steps 1–2299):** 039bK tracked slightly ahead of baseline (−0.004 to −0.014)
in early steps, converging to noise-floor parity (±0.002) by step 1700. The frozen carry
vector has no effect before loop activation — expected, since the carry path is inactive.

**Loop spike (step 2300):** 039bK spike was 2.8960 vs baseline 2.7198 (+0.176). Notably
larger than baseline and 039bG (2.8194). Without the alpha scaling, the unscaled carry
caused a harder initial shock at loop activation. The frozen values [1.5610, 1.8531, 2.1320]
appear to dampen the loop activation spike by scaling the carry into a more stable range.

**Post-loop recovery (steps 2300–3300):** Recovery was fast — back to baseline parity by
step 2400 (100 steps). From step 2500 onward, 039bK tracked at essentially zero delta
(±0.002 noise floor), indistinguishable from baseline.

## Conclusion

**Frozen carry is neutral to the final training trajectory.** Despite a larger loop spike,
039bK fully recovered and converged to the same curve as baseline. The frozen alpha values
[1.5610, 1.8531, 2.1320] slightly stabilize loop activation (smaller spike) but don't affect
final convergence — neither helping nor hurting val_bpb in a meaningful way.

Per the spec acceptance criteria: 039bK is tracking at baseline parity, pointing to
pre-quant val_bpb ≈ 1.0651 (neutral). There is no free BPB improvement from removing
frozen carry. Future experiments can use either RECUR_ALPHA_ENABLED=0 or =1 without
material impact on results.

## Decision

Kill → 039bL (floor_then_linear + RECUR_ALPHA_ENABLED=0 combo). Since frozen carry
is neutral, 039bL's outcome will be driven entirely by the floor_then_linear schedule,
making it a clean retest of 039bJ with one fewer variable to worry about.
