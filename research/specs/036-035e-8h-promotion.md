# Spec 036 — `035e` `8×H` promotion with standard `#1779` TTT

**Slug:** `035e-8h-promotion`
**Created:** 2026-04-24
**Status:** FROZEN SHELL
**Branch:** `exp/036-035e-8h-promotion`
**Commit:** `90e7b50`
**Links to:** `research/specs/035e-sparse-gate-on-1779-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `runs/035-series-report.md`

## Hypothesis

`035eA` was the strongest completed `4×H` screen in the current `035` family:

- `035eA` pre-quant `val_bpb = 1.06617649`
- better than `035A` by `-0.00061403`
- better than `035dA` by `-0.00112713`

So the next practical question is no longer whether sparse gate helps on the
screen rung. It is whether the `035e` training stack survives promotion to the
real `8×H` full pipeline with the normal `#1779` / `030` phased LoRA-TTT path.

## Baseline

Primary promotion baseline:

- `030` family `8×H` line with standard phased LoRA-TTT

Reference points:

- `030` seed `314`:
  - pre-quant `1.06821629`
  - post-TTT `1.06471941`
- `030` seed `2025`:
  - pre-quant `1.06821738`
  - post-TTT `1.06438348`
- `030` seed `777`:
  - pre-quant `1.06798687`
  - post-TTT `1.06428960`

Promotion source:

- `035eA` `4×H` pre-quant: `1.06617649`

## Config diff

Relative to the successful `035eA` `4×H` screen stack:

- promote to `8×H100`
- enable the normal `030` / `#1779` phased LoRA-TTT path
- preserve the successful `035e` sparse-gate training stack exactly
- keep `VAL_LOSS_EVERY=0` on the `8×H` promotion run

Inherited successful `035e` training-side stack:

- `MIN_LR=0.10`
- Polar NS lineage
- `FUSED_CE_ENABLED=1`
- `GPTQ_RESERVE_SECONDS=0.5`
- `VAL_LOSS_EVERY=0`
- sparse gate on, dense gated-attn off

Pinned TTT-side intent:

- `TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`
- normal `030` / `#1779` LoRA-TTT path

Pinned runnable code source:

- branch: `exp/035e-sparse-gate-on-1779-family`
- runnable code commit: `0e13ad0`

## Regime

This is a full `8×H100` promotion run.

Pinned intent:

- same model/training stack as the successful `035eA`
- same frozen recurrent `alpha/beta`
- same sparse gate path
- full quantized eval + phased LoRA-TTT

## Open parameters to pin later

Leave these explicit until we freeze the final execution contract:

- promotion seed: runtime-selectable
- approved seed shortlist: `314`, `2025`, `777`
- exact accept threshold for post-TTT: `TBD`
- whether to mirror `030` warm-start-A path exactly from its current best known branch/commit: `TBD`

These are intentionally left open for now so the promotion shell exists without
pretending the final launch contract is settled.

## Hardware ladder

0. optional smoke: `8×H100`, `2` minutes, no TTT, compile/preflight only
1. `8×H100` full pipeline, first promotion seed: `TBD`

Optional later:

2. additional seeds if the first run is competitive

## Run protocol

Optional smoke rung:

- `036-smoke`
- `8×H100`
- `MAX_WALLCLOCK_SECONDS=120`
- `TTT_ENABLED=0`
- same training stack otherwise
- discard result; use only for compile/path warmup

First promotion rung:

- `036A`
- `8×H100`
- full quantized eval + phased LoRA-TTT
- preserve the successful `035e` training stack
- seed chosen at launch from the approved shortlist

Execution rule:

- launch from `exp/035e-sparse-gate-on-1779-family`
- use the actually successful runnable code commit `0e13ad0`
- match the successful `035eA` training stack exactly
- only add the standard `030` / `#1779` full-pipeline / TTT settings
- allow execution to choose `SEED` from:
  - `314`
  - `2025`
  - `777`
- if the produced `config.json` differs on anything else, the rung is invalid

Smoke-run rule:

- if execution wants compile/path warmup first, use a separate `036-smoke` rung
- only change:
  - `MAX_WALLCLOCK_SECONDS=120`
  - `TTT_ENABLED=0`
- do not compare the smoke result against baselines
- do not treat the smoke run as a quality signal

## What to watch

- post-EMA pre-quant `val_bpb`
- quantized diagnostic `val_bpb`
- final phased-TTT `val_bpb`
- whether sparse gate remains stable and consistent under the full path

## Required artifacts

- training log
- `config.json`
- final pre-quant metrics
- final quantized diagnostic
- final phased-TTT result
- checkpoint and artifact paths needed for postmortem / replay

Smoke rung emits:

- training log
- `config.json`
- nothing from quantized eval / TTT is required

## Acceptance

To be pinned later, but the intended direction is:

- beat or at least match the stronger `030` family post-TTT results
- if competitive, promote to a multi-seed submission-style follow-up

## Notes

- This is intentionally an `8×H` promotion shell, not a frozen final run
  contract yet.
- The important thing already settled is the base branch/commit:
  `035e` should promote from the actually successful `0e13ad0` line, not the
  earlier stale spec pin.
- `VAL_LOSS_EVERY=0` is intentional for this `8×H` promotion. Do not add
  a step-4000 val checkpoint unless explicitly requested.
- the optional smoke rung is only for compile/preflight warming on the same
  `8×H` stack; it is not part of the quality comparison.
