# Spec 250E — Eval-only TTT_LOCAL_LR_MULT 0.75 → 0.65 on spec 250 outputs

**Date:** 2026-05-01 (deadline window)
**Idea:** TTT-side budget reinvestment, family member E — sub-step around #1953's chosen LR.
**Lineage:** Spec 250 outputs (V21+slope0.3) + single env var override at eval
time. **Eval-only — no retrain.**

## Hypothesis

PR #1953 reported a LR_MULT sweep on PR #1855 base over `{0.50, 0.75, 1.00, 1.25, 1.50, 2.00}` with **0.75 the winner** at val_bpb 1.06104597 (seed 42). The sweep used 0.25 step increments — **0.65 was not tested**. If the optimum
lies between 0.50 and 0.75, 0.65 is a candidate sub-step.

V21+slope0.3 also adds a training-time regularizer (slope=0.3 vs default 0.5).
A model with stronger training-time regularization may prefer a *lower* TTT
LR (less aggressive late-stage adaptation) — providing a mechanistic prior
that 0.65 could win where 0.75 was the original optimum on a less-regularized
base.

## Baseline

**Spec 250 same-seed post-TTT BPB.** Per-seed paired comparison.
Reference: spec 250 single-seed s42 = 1.05797.

## Expected Δ vs spec 250 (same seed)

**−0.0002 to +0.0003**, low confidence — could go either way.

- For: 0.65 is a finer-grain step around the 0.75 optimum. With slope=0.3
  adding regularization to training, a slightly less aggressive TTT LR
  might match the model better.
- Against: 0.75 was the empirical winner at #1855 base; absent strong
  prior to think slope=0.3 shifts the optimum, "the next step in either
  direction is worse" is more likely than "0.65 is better than 0.75".

Author estimate of P(win ≥ −0.0001 same-seed vs 250): **~25-35%**.

## Accept criteria

- **Per-seed:** post-TTT BPB at LR_MULT=0.65 ≥ −0.0001 below same-seed
  spec-250 result (lower threshold than 250B/C/D because the expected
  effect size is smaller — sub-step around chosen optimum).
- **Mean across 3 seeds:** if mean Δ ≤ −0.0003, **promote**; if
  0 ≥ mean Δ > −0.0003, **wash**; if mean Δ > 0, **kill**.
- All seeds clear 600s eval cap and 16 MB artifact cap.

## Config diff (vs spec 250)

Single env var override:

```
TTT_LOCAL_LR_MULT = 0.65     # was 0.75
```

All other env vars identical to spec 250's launch script.

Add for eval-only execution:
- `RESUME_FROM_CKPT=/path/to/runs/250-1953-leakyrelu03/seed_<N>/final_model.pt`
- `TTT_EVAL_ONLY=1`

## Code changes

**None.** Same train code as spec 250.

## Hardware ladder

- **Mini:** SKIP. Eval-only single env override.
- **Official (8×H100):** seeds 42, 0, 1234. **Eval-only.**

## Seed plan

- **Seed 42 first** for parity with 250B/C/D screening.
- If seed 42 paired Δ ≥ +0.0005 (clearly worse than 250 baseline), **abort
  0 and 1234** — sub-step in the wrong direction.
- Otherwise launch 0 and 1234 sequentially.

## Inputs

- **Required:** spec 250 outputs (same as other family members).

## Checkpoints to emit

- Per-seed `submission.json`.
- Per-seed `train.log`.

Destination: `runs/250E-ttt-lr-mult-0p65/seed_{42,0,1234}/`.

## Stop-early criteria

- **Per-seed:**
  - Eval-only run errors during model load → kill seed.
  - Eval wallclock > 595s → kill seed (cap-safety; should not happen for
    this spec since LR change doesn't affect compute).
  - Post-TTT BPB sanity floor: > 1.080 → kill seed.
- **Across seeds:** if seed 42 paired Δ ≥ +0.0005, abort 0 and 1234.

## Cost estimate

- ~$3-5 per seed (8×H100, eval-only, ~12 min/seed — same as 250 base).
- 3 seeds total: **~$9-15**.
- **No additional cap-safety overhead** vs spec 250 baseline — LR change
  is compute-neutral.

## Extra artifacts

- Per-seed `paired_delta.txt`: `seed=<N> spec250_bpb=<X> spec250E_bpb=<Y>
  delta=<Y-X>`.
- Mean Δ summary at end.

## Open questions for interview

1. **Spec 250 status.** Hard dependency.
2. **Pod region & capacity.** NE-1 first, JP fallback, 8×H100 only.
3. **Sub-step direction.** Should we also queue a 250F = LR_MULT 0.85
   (the other side of 0.75) to map out the local curve? If 250E wins,
   0.55 becomes the natural follow-on; if 250E loses, 0.85 may still
   win (asymmetric optimum). Default: don't queue 250F unless 250E
   shows signal.
4. **Submission decision.** If multiple arms win, pick lower 3-seed mean.

## Notes

- Family member of 250BCDE.
- **Cheapest spec in the family** by cap-safety budget (no extra eval
  time). Even if it loses, it costs nothing in cap headroom.
- **Most speculative win prior** — the 0.75 → 0.65 sub-step is ½ of
  PR #1953's grid resolution. If the LR curve isn't smooth (i.e. discrete
  optima), this lever does nothing.
- Strictly downstream of spec 250.
- Do not stack with 250B/C/D in this round — independent attribution
  needed.
