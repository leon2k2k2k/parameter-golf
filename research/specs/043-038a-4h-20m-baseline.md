# Spec 043 — `038A` on the `4×H100`, `1200s`, training-only rung

**Slug:** `038a-4h-20m-baseline`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `27a36a2`
**Links to:** `research/specs/038-smear-lqer-asym-8h.md`, `research/specs/042-039ba-earlier-loop-4h-20m.md`

## Hypothesis

The current comparisons mix:

- `038A` on `8×H100`, `600s`, full pipeline
- `039bA` / `042A` / `042B` on `4×H100`, training-only rungs

That is useful for directional judgment, but still mixes hardware and promotion
regimes. We need a direct `4H` baseline that preserves the `038A` architecture
so the `042` family can be compared against a same-rung `038` reference.

## Baseline

Use the `038A` stack exactly:

- frozen updated carry on
- `SMEAR_GATE_ENABLED=1`
- `LQER_ENABLED=1`
- no loop-band `penalized_tanh`
- default `LeakyReLU(0.5)^2` everywhere

Reference full promotion:

- `038A` `8×H100`, `600s`, full pipeline

This spec is the apples-to-apples `4H` counterpart:

- `4×H100`
- `1200s`
- training-only

## Config diff

Relative to the `038A` full-pipeline launch:

- `nproc_per_node: 8 -> 4`
- `MAX_WALLCLOCK_SECONDS: 600 -> 1200`
- `TTT_ENABLED: 1 -> 0`
- `TRAINING_ONLY_SCREEN: 0 -> 1`
- keep default MLP activation everywhere:
  - `MLP_OUTER_ACTIVATION=leaky_relu_square`
  - `MLP_MIDDLE_ACTIVATION=leaky_relu_square`
  - `MLP_MIDDLE_NEGATIVE_SLOPE=0.5`
  - `MLP_MIDDLE_LAYERS=3,4,5`
- keep default loop timing:
  - `ENABLE_LOOPING_AT=0.35`

Everything else should match the `038A` stack.

## Regime

This is explicitly a training-only comparison rung.

- `4×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=1200`
- `TTT_ENABLED=0`
- `TRAINING_ONLY_SCREEN=1`

Compare against:

- `039bA`
- `042A`
- `042B`
- `042C` later

## Seed policy

Use one seed only:

- `42`

## Hardware ladder

1. `4×H100`, `1200s`, training-only
2. no quantized eval
3. no TTT

## Run protocol

Single rung:

- `043A`
- `4×H100`
- `1200s`
- same `038A` architecture
- training-only stop after pre-quant diagnostic

Decision use:

- this is not a new candidate architecture
- it is the `4H` control for judging whether the `042` family is actually
  beating the `038` stack on the same rung

## Resolved base env block

Use the `038A` environment and change only the rung-control flags above.

Execution shorthand:

- start from the validated `038A` launch env
- preserve:
  - `SMEAR_GATE_ENABLED=1`
  - `LQER_ENABLED=1`
  - `RECUR_ALPHA_ENABLED=1`
  - default leaky-square MLP everywhere
- then apply:
  - `MAX_WALLCLOCK_SECONDS=1200`
  - `TTT_ENABLED=0`
  - `TRAINING_ONLY_SCREEN=1`
  - `RUN_ID=043A-038a-4h-20m-baseline`
  - `torchrun --nproc_per_node=4`

## Monitoring

Primary comparison:

- `043A` vs `042A`

Secondary comparisons:

- `043A` vs `042B`
- `043A` vs `039bA`

The point is to answer:

- does the `042` family beat a same-rung `038` control?

not:

- does it beat the old `8H` promotion directly?
