# Spec 035b — LR plateau on the original `030` alpha/beta family

**Slug:** `plateau-on-030-family`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/035b-plateau-on-030-family`
**Commit:** `TBD after freeze`
**Links to:** `research/ideas/035b-plateau-on-030-family.md`, `research/specs/035-min-lr-on-030-family.md`, `research/specs/030-025b-seed314-new-ttt.md`

## Hypothesis

If the useful schedule effect is really about the loop-onset transition rather
than the whole tail, then a plateau around loop onset should help the stronger
original `030` alpha/beta family in `4×H` screen form.

## Baseline

Use the same intended `030` `4×H` screen stack as `035`.

Pinned lineage:

- branch lineage: `exp/029-full-stack`
- runnable code line based on `c3a99b3`
- frozen `025b` carry
- `NUM_LOOPS=2`

Primary `4×H` benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

Direct schedule comparison:

- `035` (`MIN_LR=0.10`) on the same stack

## Config diff

Requires a small scheduler patch on top of the `030` family code line.

Only intended diffs from the intended `030` `4×H` screen stack:

- plateau-support code present via this branch
- `LR_PLATEAU_ENABLED=1`
- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`

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

## Plateau semantics

Use paused-time semantics:

- before plateau start: use the normal schedule
- inside plateau: hold LR constant at the schedule value at plateau start
- after plateau: resume baseline schedule at `effective_frac = frac - width`

This avoids a sharp LR cliff after plateau end.

For the first probe:

- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`

That starts exactly at loop onset and provides a direct post-kick adaptation
window.

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT

## Run protocol

First rung only:

- `035bA`
- `LR_PLATEAU_ENABLED=1`
- `LR_PLATEAU_START=0.35`
- `LR_PLATEAU_END=0.45`
- `SEED=314`

Execution rule:

- launch from `exp/035b-plateau-on-030-family`
- use the pinned runnable code commit in this spec
- match the original intended `030` `4×H` screen stack exactly
- apply only the plateau-support code lineage and the three plateau envs above
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout TBD_AFTER_FREEZE

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NA layout" >&2
  exit 1
fi

mkdir -p /workspace/runs/035b-plateau-on-030-family/run_a/seed_314
mkdir -p /tmp/torch_inductor_cache_035b_a

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/035b-plateau-on-030-family/run_a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035b_a \
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
LR_PLATEAU_ENABLED=1 LR_PLATEAU_START=0.35 LR_PLATEAU_END=0.45 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/035b-plateau-on-030-family/run_a/seed_314/train.log 2>&1
```

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log

## Sanity gate

Before accepting the result, execution must verify from `config.json` that the
only intentional diffs from the intended `030` `4×H` screen stack are:

- plateau-support code lineage
- `LR_PLATEAU_ENABLED`
- `LR_PLATEAU_START`
- `LR_PLATEAU_END`

Data-root rule:

- if the CaseOps tokenizer exists under `/workspace/data/...`, use
  `DATA_DIR=/workspace`
- if it exists under `/workspace/parameter-golf/data/...`, use
  `DATA_DIR=/workspace/parameter-golf/data`
- if neither layout exists, abort

## Accept criteria

Strong success:

- pre-quant beats `1.06770372`
- and is competitive with or better than `035`

Weak success:

- directionally positive enough to retain as the more targeted schedule variant

Failure:

- flat or worse than both the `026` `4×H` reference and `035`
