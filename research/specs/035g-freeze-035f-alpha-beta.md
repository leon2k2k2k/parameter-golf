# Spec 035g — freeze `035f`-learned `alpha/beta` back onto the `#1779` family

**Slug:** `freeze-035f-alpha-beta`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/035g-freeze-035f-alpha-beta`
**Commit:** `e8dd68e`
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

Primary freeze variant:

1. `035gA` — exact learned `alpha/beta` from `035fA`

Optional later robustness variant:

2. `035gB` — same values rounded to 4 significant digits

The purpose of `035gB` is not score-chasing. It is a robustness / simplicity
check:

- does the gain depend on exact high-precision coefficients?
- or is the learned carry geometry coarse enough that a compact rounded version
  behaves the same?

## Source values

Final learned `035fA` terminal values:

- `recur_beta` exact:
  - `[1.6940048933029175, 2.0385119915008545, 2.229182004928589]`
- `recur_alpha` exact:
  - `[[0.27734375, -0.0260009765625, 0.045654296875], [0.06787109375, -0.421875, -0.0032501220703125], [0.1123046875, 0.25390625, -0.00482177734375]]`

Rounded `035gB` freeze values (4 significant digits):

- `recur_beta` rounded:
  - `[1.694, 2.039, 2.229]`
- `recur_alpha` rounded:
  - `[[0.2773, -0.026, 0.04565], [0.06787, -0.4219, -0.00325], [0.1123, 0.2539, -0.004822]]`

Drift from the baked `025b` initialization:

- `recur_beta` init:
  - `[1.5973426, 1.8826205, 1.9906198]`
- `recur_beta` delta:
  - `[+0.0966622933, +0.1558914915, +0.2385622049]`
- `recur_alpha` init:
  - `[[0.251953125, -0.02099609375, -0.01239013671875], [0.06689453125, -0.34765625, 0.0031280517578125], [0.138671875, 0.2412109375, 0.0272216796875]]`
- `recur_alpha` delta:
  - `[[+0.025390625, -0.0050048828125, +0.05804443359375], [+0.0009765625, -0.07421875, -0.006378173828125], [-0.0263671875, +0.0126953125, -0.03204345703125]]`

`035fA` outcome reference:

- stop step: `5008`
- stop `val_bpb`: `1.0682`
- post-EMA pre-quant `val_bpb`: `1.06775175`
- quantized diagnostic `val_bpb`: `1.07712583`

`035fA` comparisons:

- vs `035eA`: `+0.00157526`
- vs `035A`: `+0.00096123`
- vs `035dA`: `+0.00044813`
- vs `026`: `+0.00004803`

These values are now the pinned `035gA` frozen coefficients.

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
2. only if `035gA` looks good, consider `035gB` with 4-significant-digit rounding

If clearly positive, then consider promotion to a fuller `8×H` run with the
normal `030` / `#1779` TTT stack.

## Run protocol

First rung:

- `035gA`
- seed `314`
- no TTT
- exact learned frozen `alpha/beta` from `035fA`

Optional later rung:

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
- `035gB` is close enough that rounding looks viable, if we choose to run it

Failure:

- learned freeze gives back the `035f` gain
- or only the unrounded exact values work and rounded values collapse badly

## Notes

- `035gA` is now the concrete main rung.
- The 4-significant-digit variant is optional and should not block the main
  exact-value follow-up.
