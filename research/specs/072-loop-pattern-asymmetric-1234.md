# Spec 072 — Loop pattern `1,2,3,3,4,5,4,5,6,5,7` (asymmetric, early-weighted) on 060A

**Status:** FROZEN — code committed at `e7ccda2` on `exp/071-loop-pattern`,
pushed to fork. Reuses 071's code; ready to run.

**Date:** 2026-04-29
**Branch:** `exp/071-loop-pattern` (shared with 071)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent:** 060A (#1855 port; forked from `exp/060-resume-ckpt @ a0a48b7`).

## Hypothesis

Sibling to 071. Where 071 tests a *symmetric tent* peaking at layer 5,
this spec tests an **asymmetric pattern weighted toward earlier layers**
(layer 3 visited 2×, layer 6 only 1×). Visit counts:

| Layer | Visits | Note |
|---|---|---|
| 0,1,2 | 1 each (encoder) | |
| **3** | **2** | weighted earlier |
| 4 | 2 | |
| **5** | **3 (peak)** | |
| 6 | 1 | |
| 7 | 1 | exit |
| 8,9,10 | 1 each (decoder) | |

Total layer-passes per step: **15** (same as 070, 071, vs 17 canonical).
Compute multiplier: 1.36×.

**Mechanistic theory:** if early layers in the loop band do feature
integration (per the loop-recurrence literature: pass 0 writes the
biggest update, late passes are smaller corrections), then *more
visits to layer 3 than to layer 6* should help — the model gets to
re-iterate the integration step but doesn't waste passes on
late-stage refinement. This is the dual hypothesis to 071's "broader
coverage centered at 5."

## Baseline

060A single-seed (1.06358 pre-quant EMA on 4×H100 matched-FLOPs).

## Expected Δ vs 060A

| Outcome | bpb | Likelihood |
|---|---|---|
| Best (early-weighted helps) | −0.0010 to 0 | low |
| Likely | +0.0005 to +0.0020 | medium |
| Worst (U-Net incoherence) | +0.0020 to +0.0040 | medium |

## Accept criteria

- **Win:** pre-quant EMA ≤ **1.06384**
- **Noise:** 1.06384 – 1.06534
- **Kill:** ≥ 1.06534

Reading 072 alongside 071: if 072 < 071 by clear margin, "early-weighted"
is the lever. If 071 < 072, "symmetric tent" is. If both ≈, the band
*shape* matters less than total visits to layer 5.

## Config diff vs 060A

```
LOOP_PATTERN = "1,2,3,3,4,5,4,5,6,5,7"
NUM_LOOPS    = 2          # any positive value enables looping_active
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.35  # unchanged
```

## Code changes

Same `exp/071-loop-pattern @ e7ccda2` as 071. No additional changes.

Index list (verified):
- `encoder_indices = [0, 1, 2, 3, 3, 4, 5]`
- `decoder_indices = [4, 5, 6, 5, 7, 8, 9, 10]`
- 15 total visits.

## Hardware ladder

- **Mini rung: SKIP** — same code path as 071; if 071's mini smoke
  passes, the LOOP_PATTERN plumbing is verified. (Re-add mini if 071
  exposes any compile issue.)
- **Single rung: 4×H100, 1 seed (42), 20-min wallclock (1200s)**,
  matched-FLOPs to 8H × 600s.

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths.

## Stop-early criteria

Same as 071.

## Cost estimate

~$2.50 (no separate mini, reuses 071's smoke result).

## Open questions for interview

1. **Sequence with 071.** Default order: run 071 first (validates plumbing),
   then 072 back-to-back on same pod (saves startup cost).
2. **Stop pod after both?** Yes by default.

## Followups gated on result

- If win: 072-promo at 8H × 600s for leaderboard-valid measurement.
- If both 071 and 072 win or noise: comparison tells us about pattern
  shape; consider a 073 hybrid.
- If both kill: closes the asymmetric-pattern hypothesis class entirely.

## See also

- `research/specs/071-loop-pattern-tent-235.md` — sibling, symmetric tent
- `research/specs/070-loop34-nl2-screen.md` — 15-pass band-shrink baseline
