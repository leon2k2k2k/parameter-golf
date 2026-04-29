# Spec 074 — Loop pattern broad staggered at matched-canonical 17 passes

**Status:** FROZEN — reuses code from `exp/071-loop-pattern @ e7ccda2`. Ready to run.

**Date:** 2026-04-29
**Branch:** `exp/071-loop-pattern` (shared with 071/072/073)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent:** 060A (#1855 port).

## Hypothesis

Sibling to 073. Both are matched-compute (17 layer-passes) shape variants
of the canonical {3,4,5}NL=2 band. Where 073 spikes layer 5 to 4 visits,
**074 keeps layer 5 at canonical's 3 visits and instead broadens
recurrence across layers 3-7** by adding visits at layers 6 and 7.

Pattern body: `1,2,3,3,4,5,4,5,6,5,6,7,7` (13 visits).
Combined with pre `[0]` + post `[8,9,10]`: **17 total layer-passes**.

| Layer | 074 visits | Canonical visits | Δ |
|---|---|---|---|
| 0,1,2 | 1 each | 1 each | 0 |
| 3 | 2 | 3 | −1 |
| 4 | 2 | 3 | −1 |
| 5 | 3 | 3 | 0 |
| **6** | **2** | 1 | **+1** |
| **7** | **2** | 1 | **+1** |
| 8,9,10 | 1 each | 1 each | 0 |

Net: shifts 2 visits from {3,4} into {6,7}. Recurrence band effectively
**extends from {3,4,5} to {3,4,5,6,7}**, but with each layer visited
fewer-but-more-spread times.

**Mechanistic theory:** if recurrence value comes from *coverage*
(touching a wider depth band, integrating features across more
processing stages) rather than *concentration* (hammering 3-5 layers),
this should beat canonical. Reads side-by-side with 073:

- 073 wins → "depth concentration" thesis (deeper visits matter more)
- 074 wins → "broad coverage" thesis (wider band matters more)
- Both lose → canonical's 3-3-3 sweet spot is genuinely optimal at 17 passes

This is the cleanest "shape vs compute" decoupling test we can run.

The encoder/decoder index lists:
- `enc=[0,1,2,3,3,4,5,4]` (8 indices)
- `dec=[5,6,5,6,7,7,8,9,10]` (9 indices)

## Baseline

060A single-seed (1.06358 pre-quant EMA on 4×H100 matched-FLOPs).

## Expected Δ vs 060A

| Outcome | bpb | Likelihood |
|---|---|---|
| Best (broad coverage helps) | −0.0010 to 0 | low-medium |
| Likely (canonical well-tuned) | −0.0003 to +0.0015 | medium |
| Worst (broader band disrupts U-Net) | +0.0015 to +0.0030 | medium-low |

## Accept criteria

- **Win:** pre-quant EMA ≤ **1.06384**
- **Noise:** 1.06384 – 1.06534
- **Kill:** ≥ 1.06534

## Config diff vs 060A

```
LOOP_PATTERN = "1,2,3,3,4,5,4,5,6,5,6,7,7"
NUM_LOOPS    = 2
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.35
```

## Code changes

Same `exp/071-loop-pattern @ e7ccda2` as 071/072/073. No additional changes.

Index list (verified):
- `encoder_indices = [0, 1, 2, 3, 3, 4, 5, 4]`
- `decoder_indices = [5, 6, 5, 6, 7, 7, 8, 9, 10]`
- 17 total visits.

## Hardware ladder

- **Mini rung: SKIP** — same code path as 071/072/073. Plumbing verified.
- **Single rung: 4×H100, 1 seed (42), 20-min wallclock (1200s)**.

Result research-side only. Promote to 8H × 600s as 074-promo if win.

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths.

## Stop-early criteria

Same as 073.

## Cost estimate

~$2.50.

## Sequencing

Run 071 first (mini smoke validates plumbing). Then on the same warm
pod run 072, 073, 074 back-to-back. Total ~$10, ~80 min wallclock.

If you only have budget for two: **073 + 074** are the matched-compute
tests, more directly comparable to canonical and to each other than
the 15-pass 071/072 set.

## Followups gated on result

- If 073 wins and 074 loses: depth concentration is the lever. Try
  layer-5 spikes with even more visits (5-visits-on-5).
- If 074 wins and 073 loses: broad coverage is the lever. Try wider
  bands (e.g. {2,3,4,5,6,7,8} with mixed visits).
- If both win at similar magnitude: any matched-compute redistribution
  beats canonical's even 3-3-3 — write a sweep spec to tune the shape.
- If both lose: canonical band shape closes; pivot to other axes.

## See also

- `research/specs/073-loop-pattern-stepped-17pass.md` — sibling, layer-5 spike
- `research/specs/071-loop-pattern-tent-235.md` — 15-pass variant (compute-cheaper)
