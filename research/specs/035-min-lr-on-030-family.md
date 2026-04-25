# Spec 035 — `MIN_LR` on the original `030` alpha/beta family

**Slug:** `min-lr-on-030-family`
**Created:** 2026-04-24
**Status:** DRAFT
**Branch:** `exp/029-full-stack`
**Commit:** `c3a99b3`
**Links to:** `research/ideas/035-min-lr-on-030-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `research/ideas/1779-next-adds-ranked.md`

## Hypothesis

`MIN_LR=0.10` is now a demonstrated positive lever on the weaker `034` branch.
If it is a real schedule improvement rather than a direct-carry-specific fix,
it should transfer to the stronger original `030` alpha/beta family.

## Baseline

Use the original `030` family as the base line:

- branch lineage: `exp/029-full-stack`
- commit: `c3a99b3`
- frozen `025b` carry
- `NUM_LOOPS=2`

For the first test, compare `4×H` to `4×H`.

Primary benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

## Config diff

No code change.

Only intended diff from the comparable `030` screen stack:

- `MIN_LR=0.10`

Everything else must remain identical to the intended `030` `4×H` screen
configuration.

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT for the first test
- `NUM_LOOPS=2`

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

If this is clearly positive, then decide whether to:

- run `0.05` / `0.15`
- or jump directly to a fuller `8×H` `030`-family result

## Run protocol

First rung only:

- `035A`
- `MIN_LR=0.10`
- `SEED=314`

Execution rule:

- match the original intended `030` `4×H` screen stack exactly
- apply exactly one env diff:
  - `MIN_LR=0.10`
- if the produced `config.json` differs on anything else, the rung is invalid

The runnable command should use a data-root preflight, not a guessed path:

- if the CaseOps tokenizer exists under `/workspace/data/...`, use
  `DATA_DIR=/workspace`
- if it exists under `/workspace/parameter-golf/data/...`, use
  `DATA_DIR=/workspace/parameter-golf/data`
- otherwise abort

## What to watch

- pre-quant post-EMA `val_bpb`
- any train instability
- any throughput regression

## Accept criteria

Strong success:

- pre-quant beats `1.06770372`

Weak success:

- directionally positive enough to justify either a `MIN_LR` ladder or a full
  `8×H` follow-up

Failure:

- flat or worse than the `026` `4×H` reference

## Notes

- This is the main active schedule-transfer test now.
- Direct-carry `035` is shelved; the stronger `030` alpha/beta line takes this
  slot instead.
