# Spec 035 — `MIN_LR` on the original `030` alpha/beta family

**Slug:** `min-lr-on-030-family`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/035-min-lr-on-030-family`
**Commit:** `c3a99b3`
**Links to:** `research/ideas/035-min-lr-on-030-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `research/ideas/1779-next-adds-ranked.md`

## Hypothesis

`MIN_LR=0.10` produced a clear win on the weaker `034` branch. If that is a
real schedule improvement rather than a direct-carry-specific fix, it should
transfer to the stronger original `030` alpha/beta family.

## Baseline

Use the original intended `030` `4×H` screen stack as the exact baseline:

- branch lineage: `exp/029-full-stack`
- runnable code commit: `c3a99b3`
- frozen `025b` carry
- `NUM_LOOPS=2`

Primary `4×H` benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

## Config diff

No code change.

Only intended diff from the original intended `030` `4×H` screen stack:

- `MIN_LR=0.10`

Everything else must remain identical, including:

- `CASEOPS_ENABLED=1`
- `TTT_ENABLED=0`
- `MLP_CLIP_SIGMAS=12.0`
- `ATTN_CLIP_SIGMAS=13.0`
- `EMBED_BITS=7`
- `EMBED_CLIP_SIGMAS=15.0`
- `MATRIX_LR=0.026`
- `GATED_ATTN_ENABLED=1`
- `GATED_ATTN_INIT_STD=0.005`
- `GATED_ATTN_QUANT_GATE=1`
- `RECUR_ALPHA_ENABLED=1`
- `NUM_LOOPS=2`
- `LOOP_START=3`
- `LOOP_END=5`
- `ENABLE_LOOPING_AT=0.35`
- `GPTQ_RESERVE_SECONDS=4`
- `GPTQ_CALIBRATION_BATCHES=16`
- `MAX_WALLCLOCK_SECONDS=1200`
- `TRAIN_LOG_EVERY=100`
- `SEED=314`

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

If this is clearly positive, then decide whether to:

- run a `MIN_LR` ladder
- or jump directly to a fuller `8×H` `030`-family follow-up

## Run protocol

First rung only:

- `035A`
- `MIN_LR=0.10`
- `SEED=314`

Execution rule:

- launch from `exp/035-min-lr-on-030-family`
- use runnable code commit `c3a99b3`
- match the original intended `030` `4×H` screen stack exactly
- apply exactly one env diff:
  - `MIN_LR=0.10`
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout c3a99b3

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NA layout" >&2
  exit 1
fi

mkdir -p /workspace/runs/035-min-lr-on-030-family/run_a/seed_314
mkdir -p /tmp/torch_inductor_cache_035_a

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/035-min-lr-on-030-family/run_a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035_a \
CASEOPS_ENABLED=1 \
TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
MIN_LR=0.10 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/035-min-lr-on-030-family/run_a/seed_314/train.log 2>&1
```

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log

## Sanity gate

Before accepting the result, execution must verify from `config.json` that the
only intentional diff from the intended `030` `4×H` screen stack is:

- `MIN_LR`

Data-root rule:

- if the CaseOps tokenizer exists under `/workspace/data/...`, use
  `DATA_DIR=/workspace`
- if it exists under `/workspace/parameter-golf/data/...`, use
  `DATA_DIR=/workspace/parameter-golf/data`
- if neither layout exists, abort

## Accept criteria

Strong success:

- pre-quant beats `1.06770372`

Weak success:

- directionally positive enough to justify either a `MIN_LR` ladder or a full
  `8×H` follow-up

Failure:

- flat or worse than the `026` `4×H` reference
