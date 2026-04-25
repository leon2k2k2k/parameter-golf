# Leaky-square backward bug — all contaminated runs were training with √0.5 in the backward

**Date:** 2026-04-25

## What we found

Every run from spec 039 onward (commit `207f38e` through `5bbf12f`) trained with
a mismatched forward/backward pair in the fused `LeakyReLU(s)²` Triton kernel.

The bug: the backward computed `2·s·x` for negative-side inputs instead of the
correct `2·s²·x`.

For our default `NEGATIVE_SLOPE=0.5`, this means:

- **Forward**: `f(x) = leaky_relu(x, 0.5)² = 0.25x²` for x < 0
- **Backward (buggy)**: gradient = `2 × 0.5 × x = x`
- **Backward (correct)**: gradient = `2 × 0.5² × x = 0.5x`

The buggy backward gradient corresponds to the correct gradient of
`leaky_relu(x, √0.5)²` — i.e. we were effectively training with slope **√0.5 ≈
0.707** in the backward while using slope **0.5** in the forward.

## Why this causes end-of-training vibration

These two don't agree. The loss surface is shaped by the forward (slope 0.5 on
the negative side), but gradient steps were computed as if the slope were 0.707.
The negative-side gradient was **2× too large** everywhere.

Early in training this probably doesn't matter much — the learning rate is large,
the model is far from a minimum, and the noise just looks like a slightly more
aggressive update. But as training enters warmdown and the learning rate
collapses, the inconsistency becomes the dominant source of gradient noise. The
optimizer is trying to converge on a loss surface shaped by 0.5, but the
gradient it follows corresponds to a different curvature. The model can't fully
settle — it keeps overshooting in the negative-activation direction. This shows
up as vibration / noisy EMA at the end of training rather than a clean plateau.

## Affected runs

All runs on the `039b` code line from `207f38e` to `5bbf12f`:

- `039b*` — loop-band activation screen
- `041A`, `041B` — penalized-tanh full runs
- `042A`, `042B`, `042C` — loop onset + LR floor screen
- `043A` — 038A faithful replay attempt

Artifacts in `runs/_contaminated_pre_5bbf12f/`.

## What is clean

Runs on the original `038` code line at commit `c8620b6` or earlier used a
hardcoded `0.5 * pre` in the backward, which is the exact correct formula for
`s=0.5` (`2 × 0.5² = 0.5`). These runs were never affected:

- spec 036, 037, 038A — all clean

`038A` (pre-quant post-EMA `val_bpb = 1.06541920`, post-TTT `1.06287`) remains
our best clean baseline.

## Fix

Commit `5bbf12f` corrects the backward to `2·s²·x`. All specs going forward are
repinned to this commit or later.
