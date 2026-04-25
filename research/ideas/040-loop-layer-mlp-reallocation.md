# Idea 040 — non-uniform MLP allocation around the looped middle layers

## Thesis

The current stack still uses a mostly uniform MLP across all physical layers,
despite the architecture already privileging the middle layers through looping.

Recent external evidence suggests this is likely suboptimal:

- **Layerwise Importance Analysis of Feed-Forward Networks in Transformer-based Language Models**
  - https://arxiv.org/abs/2508.17734
  - key result: holding total parameter count fixed, concentrating FFN capacity
    into the consecutive middle layers outperforms the standard uniform FFN
    layout

That maps unusually well onto our stack because the looped middle band is
already the part of the network that performs repeated refinement.

## Core proposal

Keep the looped middle physical layers wider:

- looped middle layers (`3,4,5`) -> `MLP_MULT=5.0`

Then compare three ways of paying for that extra width by shrinking the
non-looped layers.

Assumptions:

- `11` physical layers
- looped band is physical layers `3,4,5`
- early / encode-ish band is `0,1,2`
- late / decode-ish band is `6,7,8,9,10`
- current uniform baseline is `11 x 4.0 = 44.0` total width-units

## Three budget-matched schedules

### A. Shrink both sides evenly

- early `3.625x`
- looped middle `5.0x`
- late `3.625x`

Total width-units:

- `3 * 3.625 + 3 * 5.0 + 5 * 3.625 = 44.0`

### B. Shrink early, keep late

- early `3.0x`
- looped middle `5.0x`
- late `4.0x`

Total width-units:

- `3 * 3.0 + 3 * 5.0 + 5 * 4.0 = 44.0`

### C. Shrink late, keep early

- early `4.0x`
- looped middle `5.0x`
- late `3.4x`

Total width-units:

- `3 * 4.0 + 3 * 5.0 + 5 * 3.4 = 44.0`

So all three are effectively parameter-matched against the current uniform
`4.0x` baseline.

## Why this is interesting

- tests whether looped layers deserve extra FFN capacity at all
- tests where FFN budget is cheapest to remove from:
  - early / encode-ish
  - late / decode-ish
  - both evenly
- gives a real architectural answer rather than another global constant sweep

## Expected ranking

Current guess:

1. shrink early, keep late
2. shrink both evenly
3. shrink late, keep early

Reason: it seems more plausible that early layers can give up some FFN width
than late layers close to readout.

## 2026-04-25 update after off-spec 2xH100 screen

An off-spec `2xH100`, `600s`, training-only screen was run across:

- baseline
- `040A`
- `040B`
- `040C`

The cleanest takeaway is not that `040C` is proven. It is that this design
space is viable enough to keep pushing.

What survived triage:

- `040A` looks unattractive in the current implementation because it paid a
  real throughput penalty and underperformed.
- `040B` looks like a mild learning regression.
- `040C` did **not** clearly get worse and stayed essentially on baseline pace.

So the updated judgment is:

- **do not treat `040C` as validated**
- **do treat “widen middle, shrink late, keep early” as a plausible path**

That is already useful. It means non-uniform MLP allocation around the looped
middle band can likely be explored further without immediately damaging the
stack, and refinement should focus on the `040C` side of the family rather
than the `040A/040B` side.

## Updated direction

The next refinement direction should be:

- keep the looped middle band wider
- keep early layers near baseline
- explore gentler late-layer shrink schedules instead of the broader
  `040A/040B` families

Examples:

- early `4.0`, middle `5.0`, late `3.6`
- early `4.0`, middle `4.75`, late `3.55`
- early `4.0`, middle `4.5`, late `3.7`

The goal is no longer “which of the three coarse patterns wins?” The goal is:

- refine the `040C` family and find the best late-shrink / middle-widen tradeoff

## Main risks

- extra loop-layer width may not help enough to justify the redistribution
- early/late split may be too coarse; the real best schedule could be more
  graded
- training-only signal may not survive quantization

## Cheapest next step

Do a small training-only `4×H100` screen on the corrected `039` base code line,
holding everything fixed except the per-layer MLP width schedule.
