# Spec 041B — 1801 plus penalized-tanh full run

**Slug:** `1801-plus-penalized-tanh-8h`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/041b-1801-penalized-tanh`
**Commit:** `TBD`

## Hypothesis

The loop-band `penalized_tanh` win from `039bA` should be tested directly on
the cleaner `1801` updated-frozen-carry line, not only on the merged `038`
family.

## Runnable code source

- branch: `exp/041b-1801-penalized-tanh`
- commit: `TBD`
- script:
  [train_gpt.py](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/041b-1801-penalized-tanh/records/track_10min_16mb/2026-04-24_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE_UpdatedCarry/train_gpt.py)

## Config diff

Apply the `039bA` activation split on the `1801` code line:

- `NEGATIVE_SLOPE=0.5`
- `MLP_OUTER_ACTIVATION=leaky_relu_square`
- `MLP_MIDDLE_ACTIVATION=penalized_tanh`
- `MLP_MIDDLE_NEGATIVE_SLOPE=0.5`
- `MLP_MIDDLE_LAYERS=3,4,5`

Keep the rest of the `1801` stack unchanged.

## Regime

- `8×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=600`
- full quantization / deserialize / TTT path enabled

## Canonical env additions

```bash
NEGATIVE_SLOPE=0.5
MLP_OUTER_ACTIVATION=leaky_relu_square
MLP_MIDDLE_ACTIVATION=penalized_tanh
MLP_MIDDLE_NEGATIVE_SLOPE=0.5
MLP_MIDDLE_LAYERS=3,4,5
TTT_ENABLED=1
MAX_WALLCLOCK_SECONDS=600
```
