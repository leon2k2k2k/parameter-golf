# Spec 030b — `MIN_LR` on the original `025b` / `030` carry line

**Slug:** `min-lr-on-025b-screen`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/029-full-stack`
**Commit:** `c3a99b3`
**Links to:** `research/ideas/030b-min-lr-on-025b-screen.md`, `research/specs/030-025b-seed314-new-ttt.md`, `research/ideas/1779-next-adds-ranked.md`

## Hypothesis

The `MIN_LR=0.10` win on `034cB` may be a general schedule improvement rather
than a direct-carry-specific effect. If so, it should transfer back to the
original `025b` / `030` frozen carry line.

## Baseline

Use the original `030` family as the baseline:

- branch lineage: `exp/029-full-stack`
- commit: `c3a99b3`
- frozen `025b` carry
- `NUM_LOOPS=2`

For the first test, pre-quant is enough.

Reference numbers from the existing `030` runs:

- seed `314`: pre-quant `1.06821629`
- seed `2025`: pre-quant `1.06821738`
- seed `777`: pre-quant `1.06798687`

## Config diff

No code change.

Only intended diff from the comparable `030` screen-style stack:

- `MIN_LR=0.10`

Everything else should remain pinned to the same `025b` / `030` family.

This is a strict inheritance spec:

- match the original `030` **4×H screen** stack
- do not compare against a looser “same family” run
- only `MIN_LR` may differ

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- screen-style run
- pre-quant gate only
- no TTT required for the first test
- `NUM_LOOPS=2`
- same frozen `025b` carry lineage as `030`

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

If this is clearly positive, then decide whether to:

- run `0.05` / `0.15`
- or jump directly to a fuller `030`-family result

## Run protocol

First rung only:

- `030bA`
- `MIN_LR=0.10`
- `SEED=314`

Pinned command shape should mirror the `030` screen contract, with:

- `DATA_DIR=/workspace/parameter-golf/data`
- `TORCHINDUCTOR_CACHE_DIR=/tmp/...`
- `MIN_LR=0.10`

and otherwise inherit the intended `030` screen stack.

Execution rule:

- launch from `exp/029-full-stack` at `c3a99b3`
- mirror the original `030` 4×H screen config exactly
- apply exactly one env diff:
  - `MIN_LR=0.10`
- if `config.json` differs from the intended `030` screen config on anything
  else, the run is invalid

Pinned command shape:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout c3a99b3

TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_030b_a \
DATA_DIR=/workspace/parameter-golf/data \
ARTIFACT_DIR=/workspace/runs/030b-min-lr-on-025b-screen/run_a/seed_314 \
CASEOPS_ENABLED=1 \
TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
MIN_LR=0.10 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## What to watch

- pre-quant post-EMA `val_bpb`
- any train instability
- any obvious throughput regression

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log

## Sanity gate

Before accepting the result, execution must verify from `config.json` that the
only intentional diff from the original `030` 4×H screen contract is:

- `MIN_LR`

## Accept criteria

Strong success:

- pre-quant clearly beats the usual seed-314 `~1.068216`

Weak success:

- directionally positive enough to justify either the full ladder or a full
  `030`-family follow-up

Failure:

- flat or worse against the normal seed-314 pre-quant level

## Notes

- This is intentionally the cheap transfer test.
- It is meant to answer whether `MIN_LR` belongs back on the original frozen
  carry line, not to replace the full `030` pipeline immediately.
