# Spec 035h — learnable `alpha/beta` on the sparse-gate `035e` family

**Slug:** `learnable-alpha-beta-on-sparse-gate-family`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/035h-learnable-alpha-beta-on-sparse-gate-family`
**Commit:** `6d0c37c`
**Links to:** `research/specs/035e-sparse-gate-on-1779-family.md`, `research/specs/035f-learnable-alpha-beta-on-1779-family.md`

## Hypothesis

`035fA` showed that the current stack wants to move recurrent `alpha/beta`, but
that run was on the non-sparse-gate family and did not beat the stronger sparse
`035eA` branch. So the right test is not to mutate `035f`; it is to ask whether
learnable `alpha/beta` helps specifically on top of the best current `035e`
training stack.

This spec is exactly:

- start from the successful `035e` sparse-gate branch
- keep the full `035e` training-side stack
- make recurrent `alpha/beta` learnable during normal training
- first test on the same `4×H` no-TTT screen rung

## Baseline

Primary baseline:

- `035eA` pre-quant `val_bpb = 1.06617649`

Other references:

- `035A` pre-quant `1.06679052`
- `035gA` pre-quant `1.06711750`
- `026` pre-quant `1.06770372`

## Config diff

Relative to the successful `035eA` stack:

- preserve sparse gate
- preserve `MIN_LR=0.10`
- preserve Polar NS lineage
- preserve `FUSED_CE_ENABLED=1`
- preserve `GPTQ_RESERVE_SECONDS=0.5`
- preserve `VAL_LOSS_EVERY=0`
- new diff: `RECUR_ALPHA_BETA_LEARNABLE=1`
- neutral recurrent init:
  - `recur_beta = [1.0, 1.0, 1.0]`
  - `recur_alpha = 0`

## Regime

Use a `4×H100` screen-only rung.

- no TTT
- pre-quant gate only
- `NUM_LOOPS=2`

## Run protocol

First rung only:

- `035hA`
- seed `314`
- no TTT
- sparse gate on
- learnable recurrent `alpha/beta`

Execution rule:

- launch from `exp/035h-learnable-alpha-beta-on-sparse-gate-family`
- use the pinned runnable commit from this branch
- match the successful `035eA` stack exactly
- only add:
  - `RECUR_ALPHA_BETA_LEARNABLE=1`
  - neutral recurrent init in code lineage
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 6d0c37c

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NE/NA layout" >&2
  exit 1
fi

mkdir -p /workspace/runs/035h-learnable-alpha-beta-on-sparse-gate-family/run_a/seed_314
mkdir -p /tmp/torch_inductor_cache_035h_a

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/035h-learnable-alpha-beta-on-sparse-gate-family/run_a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035h_a \
CASEOPS_ENABLED=1 \
TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0 \
RECUR_ALPHA_ENABLED=1 RECUR_ALPHA_BETA_LEARNABLE=1 \
NUM_LOOPS=2 \
LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
MUON_BACKEND_STEPS=5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
VAL_LOSS_EVERY=0 \
FUSED_CE_ENABLED=1 \
MIN_LR=0.10 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/035h-learnable-alpha-beta-on-sparse-gate-family/run_a/seed_314/train.log 2>&1
```

## What to watch

- pre-quant post-EMA `val_bpb`
- whether learnable `alpha/beta` now helps on the stronger sparse-gate branch
- alpha/beta drift and gradient norms
- train stability

## Accept criteria

Strong success:

- beats `035eA`

Useful partial success:

- beats `035A` and `035gA` but not `035eA`

Failure:

- flat/worse than `035eA`
- or unstable training
