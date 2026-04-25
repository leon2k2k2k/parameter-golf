# Spec 035b — LR plateau on the original `030` alpha/beta family

**Slug:** `plateau-on-030-family`
**Created:** 2026-04-24
**Status:** DRAFT
**Branch:** `exp/029-full-stack`
**Commit:** `c3a99b3`
**Links to:** `research/ideas/035b-plateau-on-030-family.md`, `research/specs/035-min-lr-on-030-family.md`, `research/specs/030-025b-seed314-new-ttt.md`

## Hypothesis

If the useful schedule effect is really about the loop-onset transition rather
than the whole tail, then a plateau around loop onset should help the stronger
`030` alpha/beta family in `4×H` screen form.

## Baseline

Use the same intended `030` `4×H` screen stack as `035`.

Primary benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

Direct schedule comparison:

- `035` (`MIN_LR=0.10`) once available

## Config diff

Requires a small scheduler patch, analogous to the earlier `034dA` idea.

Only intended schedule diffs:

- `LR_PLATEAU_ENABLED=1`
- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`

Everything else should match the intended `030` `4×H` screen stack.

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT for the first test
- `NUM_LOOPS=2`

## Plateau semantics

Use paused-time semantics:

- before start: baseline schedule
- inside plateau: constant at the schedule value at plateau start
- after plateau: resume baseline schedule at `effective_frac = frac - width`

That avoids a sharp LR cliff after plateau end.

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

## Run protocol

First rung only:

- `035bA`
- `LR_PLATEAU_ENABLED=1`
- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`
- `SEED=314`

Like `035`, the final runnable command should:

- match the intended `030` `4×H` screen stack exactly
- use a data-root probe for JP vs NA
- invalidate the rung if anything besides the plateau change drifts

## What to watch

- pre-quant post-EMA `val_bpb`
- any train instability
- any throughput regression

## Accept criteria

Strong success:

- beats `1.06770372`
- and is competitive with or better than `035`

Weak success:

- directionally positive, worth retaining as the more targeted schedule variant

Failure:

- flat or worse than both the `026` `4×H` reference and `035`
