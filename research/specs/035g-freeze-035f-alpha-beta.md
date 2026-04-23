# Spec 035g — freeze `035f`-learned `alpha/beta` back onto the `#1779` family

**Slug:** `freeze-035f-alpha-beta`
**Created:** 2026-04-24
**Status:** BLOCKED ON `035fA` PARAMETERS
**Branch:** `exp/035g-freeze-035f-alpha-beta`
**Commit:** `d151697`
**Links to:** `research/ideas/035g-freeze-035f-alpha-beta.md`, `research/specs/035f-learnable-alpha-beta-on-1779-family.md`, `research/specs/030-025b-seed314-new-ttt.md`

## Hypothesis

If `035fA` learns a better recurrent `alpha/beta` setting than the old frozen
`025b` values, the simplest high-value follow-up is to freeze those learned
coefficients back into the normal `#1779` / `030` family and see whether the
gain survives without training-time learnability.

This is the clean “measure, then refreeze” follow-up:

- `035f` asks whether the current stack wants different `alpha/beta`
- `035g` asks whether those learned values are themselves a better frozen carry

## Baseline

Primary baseline:

- `035A` pre-quant `val_bpb = 1.06679052`

Other family references:

- `035eA` pre-quant `val_bpb = 1.06617649`
- `026` screen seed `314`: pre-quant `1.06770372`

## Config diff

Relative to the intended `030` `4×H` screen stack:

- keep recurrent `alpha/beta` frozen during training
- replace the baked `025b` values with the learned terminal `035fA` values
- no TTT for the first test

Two freeze variants are allowed once `035fA` finishes:

1. `035gA` — exact learned `alpha/beta` from `035fA`
2. `035gB` — same values rounded to 4 significant digits

The purpose of `035gB` is not score-chasing. It is a robustness / simplicity
check:

- does the gain depend on exact high-precision coefficients?
- or is the learned carry geometry coarse enough that a compact rounded version
  behaves the same?

## Source values

Blocked until `035fA` completes.

When `035fA` finishes, pin:

- final learned `recur_beta`
- final learned `recur_alpha`
- before/after drift from the `025b` initialization

The spec should not be promoted to READY until those values are copied into the
branch-local spec and code. This draft may live on a real remote branch early,
but it is not runnable until the learned coefficients are pinned.

## Regime

Use a `4×H100` screen-only rung first.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT for the first test
- `NUM_LOOPS=2`

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

Order:

1. run `035gA` first using exact learned values
2. only if `035gA` looks good, run `035gB` with 4-significant-digit rounding

If clearly positive, then consider promotion to a fuller `8×H` run with the
normal `030` / `#1779` TTT stack.

## Run protocol

First rung:

- `035gA`
- seed `314`
- no TTT
- exact learned frozen `alpha/beta` from `035fA`

Optional second rung:

- `035gB`
- same stack
- same seed
- same learned values, but rounded to 4 significant digits before freezing

Execution rule:

- inherit the intended `030` `4×H` screen stack exactly
- apply only the frozen-`alpha/beta` replacement
- if `035gB` is used, the only extra diff is rounding the copied values to
  4 significant digits
- if the produced `config.json` differs on anything else, the rung is invalid

## What to watch

- pre-quant post-EMA `val_bpb`
- whether exact learned freeze beats the old frozen `025b` baseline
- whether the rounded variant stays close to the exact one
- whether the learned values look coherent or noisy

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log
- explicit frozen `alpha/beta` values in the spec and run notes

## Accept criteria

Strong success:

- `035gA` beats `035A`
- and ideally stays near or beats `035eA`

Useful partial success:

- `035gA` improves on `026` and the old frozen `025b` line
- `035gB` is close enough that rounding looks viable

Failure:

- learned freeze gives back the `035f` gain
- or only the unrounded exact values work and rounded values collapse badly

## Notes

- This spec is intentionally contingent on `035fA`.
- Do not freeze or run it until the actual learned coefficients are available.
- The 4-significant-digit variant is optional and should not block the main
  exact-value follow-up.
