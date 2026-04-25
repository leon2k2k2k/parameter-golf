# Spec 041B — 1797-style Smear+LQER plus loop-band penalized-tanh `8×H`

**Slug:** `1797-plus-penalized-tanh-8h`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/041b-1797-penalized-tanh`
**Commit:** `26eb119`

## Hypothesis

The strong `039bA` loop-band `penalized_tanh` signal should be tested on a
`#1797`-style base:

- `SmearGate` on
- `LQER-asym` on
- **no frozen carry**

In this checkout, the faithful way to do that is to use the same runnable
Smear+LQER code line as `039b`, but explicitly disable the frozen `recur_alpha`
carry by env.

## Runnable code source

- branch: `exp/041b-1797-penalized-tanh`
- commit: `26eb119`
- script:
  [train_gpt.py](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/041b-1797-penalized-tanh/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py)

Smoke preflight files on the runnable branch:

- [041b-1797-plus-penalized-tanh-2h-smoke.env](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/041b-1797-penalized-tanh/research/specs/041b-1797-plus-penalized-tanh-2h-smoke.env)
- [041b-1797-plus-penalized-tanh-2h-smoke.sh](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/041b-1797-penalized-tanh/research/specs/041b-1797-plus-penalized-tanh-2h-smoke.sh)

## Config diff

Relative to the `041A` / `039bA` full-run setup:

- keep:
  - `SMEAR_GATE_ENABLED=1`
  - `LQER_ENABLED=1`
  - `LQER_RANK=4`
  - `LQER_TOP_K=3`
  - `LQER_FACTOR_BITS=4`
  - `LQER_ASYM_ENABLED=1`
  - `LQER_ASYM_GROUP=64`
  - `MLP_OUTER_ACTIVATION=leaky_relu_square`
  - `MLP_MIDDLE_ACTIVATION=penalized_tanh`
  - `MLP_MIDDLE_LAYERS=3,4,5`
- change:
  - `RECUR_ALPHA_ENABLED=0`

So `041B` is the no-frozen-carry sibling of `041A`.

## Regime

- `8×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=600`
- full quantization / deserialize / TTT path enabled
- `TRAIN_LOG_EVERY=100`

## Launch

Use the pushed branch and env in the branch-local spec or smoke file as needed.

Canonical full-run branch:

- `fork/exp/041b-1797-penalized-tanh`

## Verification notes

Verified on the runnable code branch:

- loop-band activation API is present:
  - `MLP_OUTER_ACTIVATION`
  - `MLP_MIDDLE_ACTIVATION`
  - `MLP_MIDDLE_LAYERS`
- `penalized_tanh` implementation is present
- `LQER_ENABLED` is present
- `RECUR_ALPHA_ENABLED` is present and can disable the frozen carry path
