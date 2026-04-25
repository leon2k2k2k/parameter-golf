# Spec 035f — learnable `alpha/beta` on the `#1779` / `030` family

**Slug:** `learnable-alpha-beta-on-1779-family`
**Created:** 2026-04-24
**Status:** DRAFT
**Branch:** `exp/029-full-stack`
**Commit:** `c3a99b3`
**Links to:** `research/ideas/035f-learnable-alpha-beta-on-1779-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `research/specs/035-min-lr-on-030-family.md`

## Hypothesis

Our current `#1779` / `030` family freezes the recurrent `alpha/beta` carry at
the old `025b` values. That is simple and strong, but it may leave performance
on the table if the current stack wants to recalibrate those coefficients under
the modern CaseOps / gated-attn / quant-gate regime.

So this spec asks:

- keep the same `030` family
- but make recurrent `alpha/beta` trainable during normal training
- and test that first as a `4×H` screen

## Baseline

Use the intended `030` `4×H` screen stack as the exact base line.

Primary `4×H` reference:

- `026` screen seed `314`: pre-quant `1.06770372`

Internal references:

- `035`: same stack + `MIN_LR=0.10`
- `030` screen intent: frozen `025b` carry, no TTT

## Config diff

Relative to the intended `030` `4×H` screen stack:

- recurrent `alpha/beta` becomes learnable during training
- initialize from the same `025b` values currently baked into the line
- no TTT for the first test

The initial values should remain exactly:

- `recur_beta = [1.5973426, 1.8826205, 1.9906198]`
- `recur_alpha = [[0.251953125, -0.02099609375, -0.01239013671875], [0.06689453125, -0.34765625, 0.0031280517578125], [0.138671875, 0.2412109375, 0.0272216796875]]`

This is a **relearn from strong init**, not a random reinit and not a neutral
init.

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT for the first test
- `NUM_LOOPS=2`

## Why this is interesting

This is the direct “do we still want frozen carry?” test on the strongest
alpha/beta family.

It is not about TTT adaptation and not about direct-carry. It is simply:

- keep the old strong carry structure
- let the coefficients move again under the current stack

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

If clearly positive, then decide whether to:

- try `MIN_LR` on top
- or promote directly to a fuller `8×H` run

## Run protocol

First rung only:

- `035fA`
- seed `314`
- no TTT
- learnable recurrent `alpha/beta`

Execution rule:

- match the intended `030` `4×H` screen stack exactly
- apply only the learnable-`alpha/beta` code / env diffs
- initialize from the existing baked `025b` values
- if the produced `config.json` differs on anything else, the rung is invalid

## What to watch

- pre-quant post-EMA `val_bpb`
- whether `alpha/beta` drift materially or stay near the frozen values
- any train instability
- whether the line beats the old `026` `4×H` reference

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log
- explicit `alpha/beta` before/after logging
- periodic drift logging if implemented

## Accept criteria

Strong success:

- pre-quant clearly beats `1.06770372`
- and the learned coefficients move in a coherent way

Weak success:

- roughly tied result, but with informative coefficient motion worth following up

Failure:

- flat/worse with no meaningful movement
- or unstable training

## Notes

- This is a research draft only.
- It still needs a real code branch:
  - enable trainable recurrent `alpha/beta`
  - preserve exact `025b` initialization
  - add logging
