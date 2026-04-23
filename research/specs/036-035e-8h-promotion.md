# Spec 036 — sparse-gate `8×H` promotion with updated rounded `alpha/beta`

**Slug:** `sparse-updated-alpha-beta-8h-promotion`
**Created:** 2026-04-24
**Status:** BLOCKED ON `035h` PARAMETERS
**Branch:** `exp/036-035e-8h-promotion`
**Commit:** `TBD after `035h` completes`
**Links to:** `research/specs/035e-sparse-gate-on-1779-family.md`, `research/specs/035h-learnable-alpha-beta-on-sparse-gate-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `runs/035-series-report.md`

## Hypothesis

`035eA` was the strongest completed `4×H` screen in the current `035` family.
The next intended promotion is therefore the sparse-gate stack at `8×H`, but
using the updated recurrent carry from the newer sparse learnable-alpha/beta
run `035h`, not the older non-sparse learned carry. This spec stays blocked
until those final `035h` parameters are available.

## Baseline

Primary promotion baselines:

- `030` family `8×H` line with standard phased LoRA-TTT
- `035eA` `4×H` sparse-gate screen at `1.06617649`

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

Updated rounded recurrent carry to promote:

- source: final `035h` learned `alpha/beta`
- rounding rule: **2 decimal places**, not 2 significant digits
- exact values: `TBD after 035h completes`

Older non-authoritative example from the earlier non-sparse `035f` line:

- `beta = [1.69, 2.04, 2.23]`
- `alpha = [[0.28, -0.03, 0.05], [0.07, -0.42, -0.00], [0.11, 0.25, -0.00]]`

These example values are here only to pin the intended rounding convention.
They are not the values to run unless `035h` independently lands on them.

## Config diff

Relative to the successful `035eA` `4×H` screen stack:

- promote to `8×H100`
- enable the normal `030` / `#1779` phased LoRA-TTT path
- preserve the successful `035e` sparse-gate training stack
- replace the old baked `025b` recurrent carry with the updated rounded frozen
  carry from the later sparse learnable-alpha/beta line `035h`
- keep `VAL_LOSS_EVERY=0` on the `8×H` promotion run

Inherited successful `035e` training-side stack:

- `MIN_LR=0.10`
- Polar NS lineage
- `FUSED_CE_ENABLED=1`
- `GPTQ_RESERVE_SECONDS=0.5`
- `VAL_LOSS_EVERY=0`
- sparse gate on, dense gated-attn off
- recurrent carry still frozen, but using the final rounded `035h` values once
  they are available

Pinned TTT-side intent:

- `TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`
- normal `030` / `#1779` LoRA-TTT path

Pinned runnable code source:

- shell/spec branch: `exp/036-035e-8h-promotion`
- future runnable code branch: `TBD after 035h completes`
- future runnable code commit: `TBD after 035h completes`

## Regime

This is a full `8×H100` promotion run.

Pinned intent:

- same model/training stack as the successful `035eA`
- sparse gate path from `035e`
- updated rounded frozen recurrent `alpha/beta` from `035h`
- full quantized eval + phased LoRA-TTT

## Seed policy

Promotion seed is runtime-selectable from the approved shortlist:

- `1`
- `777`
- `2025`

Recommended first seed:

- `1`

Execution may choose any one of the approved seeds at launch, but must record
the chosen seed in `notes.md` and `config.json`.

## Hardware ladder

0. optional smoke: `8×H100`, `2` minutes, no TTT, compile/preflight only
1. `8×H100` full pipeline, first promotion seed from the approved shortlist

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
- preserve the successful `035e` sparse-gate stack
- use the updated rounded frozen recurrent carry from final `035h`
- seed chosen at launch from the approved shortlist

Execution rule after unblocking:

- launch from the future `036` runnable code branch derived from `035e`
- freeze in the final rounded `035h` recurrent carry
- keep the successful `035eA` sparse-gate stack otherwise
- add the standard `030` / `#1779` full-pipeline / TTT settings
- allow execution to choose `SEED` from:
  - `1`
  - `777`
  - `2025`
- require `config.json`
- if the produced config drifts on anything except the intended updated frozen
  recurrent carry, the rung is invalid

Pinned smoke/full commands:

- `TBD after 035h completes`

## Acceptance

This run is interesting if it is competitive with the strong `030` post-TTT
range and clearly validates the sparse-gate plus updated-carry promotion path.

Primary target:

- final post-TTT `val_bpb` competitive with the better `030` seeds

Secondary target:

- no sign that the updated rounded carry breaks the successful sparse-gate line

## Why blocked

The current open issue is not the `8×H` protocol itself. It is the source of
the promoted recurrent carry:

- we want final `035h` sparse learned `alpha/beta`
- rounded to 2 decimal places
- not the older non-sparse `035f` values
- and not the earlier mistaken 2-significant-digit version
