# Spec 035f — learnable `alpha/beta` on the `#1779` / `030` family

**Slug:** `learnable-alpha-beta-on-1779-family`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/035f-learnable-alpha-beta-on-1779-family`
**Commit:** `b0cb1bc`
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

- launch from `exp/035f-learnable-alpha-beta-on-1779-family`
- use runnable code commit `b0cb1bc`
- match the intended `030` `4×H` screen stack exactly
- apply only the learnable-`alpha/beta` code / env diffs
- initialize from the existing baked `025b` values
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout b0cb1bc

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NE/NA layout" >&2
  exit 1
fi

mkdir -p /workspace/runs/035f-learnable-alpha-beta-on-1779-family/run_a/seed_314
mkdir -p /tmp/torch_inductor_cache_035f_a

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/035f-learnable-alpha-beta-on-1779-family/run_a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035f_a \
CASEOPS_ENABLED=1 \
TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 RECUR_ALPHA_BETA_LEARNABLE=1 \
NUM_LOOPS=2 \
LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/035f-learnable-alpha-beta-on-1779-family/run_a/seed_314/train.log 2>&1
```

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

## Sanity gate

Before accepting the result, execution must verify from `config.json` that the
only intentional diffs from the intended `030` `4×H` screen stack are:

- learnable-`alpha/beta` code lineage
- `RECUR_ALPHA_BETA_LEARNABLE=1`

and that the initial values still match the baked `025b` coefficients.

Data-root rule:

- if the CaseOps tokenizer exists under `/workspace/data/...`, use
  `DATA_DIR=/workspace`
- if it exists under `/workspace/parameter-golf/data/...`, use
  `DATA_DIR=/workspace/parameter-golf/data`
- if neither layout exists, abort

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

- This spec preserves exact `025b` initialization and only changes whether
  those coefficients are trainable.
