# Spec 071 — Loop pattern `2,3,4,5,4,5,6,5,6,7` ("tent peak at 5") on 060A

**Status:** FROZEN — code committed at `e7ccda2` on `exp/071-loop-pattern`,
pushed to fork. Ready to run.

**Date:** 2026-04-29
**Branch:** `exp/071-loop-pattern`
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent:** 060A (#1855 port; forked from `exp/060-resume-ckpt @ a0a48b7`).

## Hypothesis

The 040/041/070 series proved that any 2-layer subset of the canonical
{3,4,5} band loses ~+0.0023 vs canonical, regardless of which layer is
dropped (070 dropped layer 5; 041H dropped layer 3, both lost similarly).
This rules out simple band-shrinking as a lever and suggests the
canonical {3,4,5} captures something specific to the band shape.

The remaining open question on the loop-band axis: **does a non-contiguous
or shape-shifted pattern beat the canonical contiguous {3,4,5} band?**
Zero PRs in the public timeline have tested this.

This spec tests a **tent-shaped pattern**: visits layers 2-7 with
emphasis concentrated at layer 5. Visit counts:

| Layer | Visits |
|---|---|
| 0,1,2,3 | 1 each (encoder) |
| 4 | 2 |
| **5** | **3 (peak)** |
| 6 | 2 |
| 7 | 1 (exit) |
| 8,9,10 | 1 each (decoder) |

Total layer-passes per step: **15** (same as 070, vs 17 for canonical).
Compute multiplier: 1.36× (12% lower than canonical's 1.55×).

**Mechanistic theory:** the canonical {3,4,5} band hits the same 3 layers
3× each. The tent pattern instead spreads visits across a wider depth
range (layers 2-7) while preserving a "deepest concentration" at layer 5.
If iterative refinement benefits from *broader depth coverage* rather
than *concentrated repetition*, this should win. If the canonical band
benefits specifically from hammering layers 3-5, this should lose.

## Baseline

060A single-seed (1.06358 pre-quant EMA on 4×H100 matched-FLOPs).

## Expected Δ vs 060A

| Outcome | bpb | Likelihood |
|---|---|---|
| Best (broader coverage wins) | −0.0010 to 0 | low |
| Likely (between 070 and canonical) | +0.0005 to +0.0020 | medium |
| Worst (U-Net skip incoherence dominates) | +0.0020 to +0.0040 | medium |

## Accept criteria

- **Win:** pre-quant EMA ≤ **1.06384** (within 0.00026 of 060A baseline)
- **Noise:** 1.06384 – 1.06534
- **Kill:** ≥ 1.06534

## Config diff vs 060A

```
LOOP_PATTERN = "2,3,4,5,4,5,6,5,6,7"
NUM_LOOPS    = 2          # kept positive; any value > 0 enables looping_active
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.35  # unchanged
```

## Code changes

`exp/071-loop-pattern @ e7ccda2`. Adds `LOOP_PATTERN` env var that, when
non-empty, overrides the contiguous-band construction with an explicit
comma-separated layer-index sequence. Pre/post layers fill in to
NUM_LAYERS automatically. Empty default = existing logic unchanged.

Validation: rejects empty patterns and out-of-range indices.

Index list (verified):
- `encoder_indices = [0, 1, 2, 3, 4, 5, 4]`
- `decoder_indices = [5, 6, 5, 6, 7, 8, 9, 10]`
- 15 total visits.

## Hardware ladder

- **Mini rung: 4×H100, 5-min smoke** — required (code change). Verify
  no compile pathology, looping activates correctly, encoder/decoder
  index lists in train.log match expected.
- **Single rung: 4×H100, 1 seed (42), 20-min wallclock (1200s)**,
  matched-FLOPs to 8H × 600s via GRAD_ACCUM=2.

Result is research-side only (not leaderboard-valid due to 4H >600s).
If 071 wins, promote to 8H × 600s as 071-promo for valid measurement.

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths.

## Stop-early criteria

- train_loss > 5.0 at step 1000 → kill
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- mid-run torch.compile recompile → kill (always-tensor violated)
- LOOP_PATTERN parse error → kill, fix env var
- NaN → kill

## Cost estimate

~$1 mini + ~$2.50 official = **~$3.50**.

## Open questions for interview

1. **Smoke check first.** Mini rung is required because this is a code
   change, even though the change is __init__-time only. Confirm
   `layer_loop:enabled encoder:[0,1,2,3,4,5,4] decoder:[5,6,5,6,7,8,9,10]`
   appears in train.log.
2. **Stop pod after?** Yes — but 072 is the immediate followup; if 072
   is queued back-to-back, keep pod warm.

## Followups gated on result

- If win: 071-promo at 8H × 600s for leaderboard-valid measurement.
- If win or noise: 072 already specced (different pattern shape).
- If kill: closes the asymmetric-pattern hypothesis. Tent-shaped
  patterns documented as not beating canonical.

## See also

- `research/specs/072-loop-pattern-asymmetric-12334545657.md` — sibling
  spec, different pattern shape (asymmetric, weighted earlier)
- `research/specs/070-loop34-nl2-screen.md` — 15-pass band-shrink test
  (lost +0.0024)
- `research/timelines/loop-recurrence-2026-04-27.md` — full
  recurrence-axis empirical timeline
