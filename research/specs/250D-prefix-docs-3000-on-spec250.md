# Spec 250D — Eval-only PHASED_TTT_PREFIX_DOCS 2500 → 3000 on spec 250 outputs

**Date:** 2026-05-01 (deadline window)
**Idea:** TTT-side budget reinvestment, family member D — aggressive prefix bump.
**Lineage:** Spec 250 outputs (V21+slope0.3) + single env var override at eval
time. **Eval-only — no retrain.**

## Hypothesis

Larger prefix bump than 250B (2750). Tests whether the V21 TTT regime can
absorb a +20% prefix increase before the wallclock cap or signal saturates.
PR #1925's claim was that 3500 docs on #1855 base was essentially neutral.
This spec is the closest in-budget approximation to that on V21+slope0.3 —
3000 docs is the largest value that *might* fit inside the 600s eval cap,
with very thin margin.

If 250B (2750) wins, this is the natural follow-on push. If 250B washes,
this likely also washes. **The reason this is a spec rather than a wait-
and-see follow-on is to maximize compute parallelism on deadline day** —
running 250B and 250D on different seeds simultaneously surfaces the
prefix curve in one pass.

## Baseline

**Spec 250 same-seed post-TTT BPB.** Per-seed paired comparison.
Reference: spec 250 single-seed s42 = 1.05797.

## Expected Δ vs spec 250 (same seed)

**−0.0001 to −0.0004**, low confidence.

- For: same as 250B but more aggressive. If TTT signal scales with prefix
  past 2500, 3000 captures more of it.
- Against: same as 250B plus diminishing returns at higher prefix counts.
- **Cap-safety risk dominates the EV calculation.** Even if BPB lands at
  the optimistic end, a single seed breaching 600s disqualifies it.

Author estimate of P(win ≥ −0.0002 same-seed vs 250 AND eval_time ≤ 595s
on all 3 seeds): **~15-25%**. Lower than 250B because the conjunction with
cap-safety reduces the probability mass.

## Accept criteria

- **Per-seed:** post-TTT BPB at PREFIX_DOCS=3000 ≥ −0.0002 below same-seed
  spec-250 result.
- **Cap safety:** **all seeds must clear 600s eval cap with at least 5s
  margin** (i.e., eval_time ≤ 595s). If any seed breaches 595s, that seed
  is disqualified for submission even if BPB is good.
- **Mean across qualifying seeds:** if mean Δ ≤ −0.0003, **promote**;
  if 0 ≥ mean Δ > −0.0003, **wash**; if mean Δ > 0, **kill**.
- All seeds clear 16 MB artifact cap.

## Config diff (vs spec 250)

Single env var override:

```
PHASED_TTT_PREFIX_DOCS = 3000     # was 2500
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

- **Seed 42 first** as smoke-and-budget-check. **Cap-safety verification
  is the primary screening criterion.** If seed 42 eval_time > 595s,
  abort 0 and 1234 even if BPB looks good — wallclock variance on
  subsequent seeds will breach the cap.
- If seed 42 paired Δ ≥ +0.0005 OR eval_time > 595s, **abort 0 and 1234**.
- Otherwise launch 0 and 1234 sequentially.

## Inputs

- **Required:** spec 250 outputs (same as 250B/C).

## Checkpoints to emit

- Per-seed `submission.json`.
- Per-seed `train.log`.

Destination: `runs/250D-prefix-docs-3000/seed_{42,0,1234}/`.

## Stop-early criteria

- **Per-seed:**
  - Eval-only run errors during model load → kill seed.
  - Eval wallclock > 595s during the TTT phased eval → kill seed
    immediately (cap breach imminent — primary risk for this spec).
  - Post-TTT BPB sanity floor: > 1.080 → kill seed.
- **Across seeds:** if seed 42 paired Δ ≥ +0.0005 OR eval_time > 595s,
  abort 0 and 1234.

## Cost estimate

- ~$3-5 per seed (8×H100, eval-only, ~13 min/seed).
- 3 seeds total: **~$9-15**.

## Extra artifacts

- Per-seed `paired_delta.txt`: `seed=<N> spec250_bpb=<X> spec250D_bpb=<Y>
  delta=<Y-X> spec250D_eval_time_s=<T>`.
- **Eval-time max across seeds is the dominant gate** for this spec.

## Open questions for interview

1. **Spec 250 status.** Hard dependency.
2. **Pod region & capacity.** NE-1 first, JP fallback, 8×H100 only.
3. **Cap-safety threshold.** 595s is the default safety floor. Tighter
   (590s) gives more margin for seed-to-seed variance; looser (598s)
   trades risk for slightly higher headroom usage. Default 595s.
4. **Seed-42 abort sensitivity.** Does seed 42's eval_time at exactly 595s
   trigger abort, or do we wait for seed 0 to confirm? Default: abort
   immediately at 595s. Wallclock variance is typically ±5-10s; surviving
   one seed at 595s with 5s margin gives ≈30% chance of breaching on a
   subsequent seed.
5. **Submission decision.** If multiple arms in the family win, pick
   lower 3-seed mean.

## Notes

- Family member of 250BCDE.
- **Higher cap-safety risk than 250B and 250C** — only 6s margin to 600s
  cap on the budget estimate (~80s expected eval cost on top of 514s
  base = ~594s, almost exactly at the cap). Empirically run-to-run
  variance can swing this either way.
- **If 250B wins and 250D is wash-or-cap-breached**, the prefix curve
  has its useful range bounded at 2750-3000.
- Strictly downstream of spec 250.
- Do not stack with 250B/C in this round — independent attribution
  needed.
