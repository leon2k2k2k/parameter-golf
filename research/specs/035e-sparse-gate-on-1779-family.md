# Spec 035e — sparse-gate follow-up on the `#1779` / `030` alpha/beta family

**Slug:** `sparse-gate-on-1779-family`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/035e-sparse-gate-on-1779-family`
**Commit:** `d205ff5`
**Links to:** `research/ideas/035e-sparse-gate-on-1779-family.md`, `research/specs/035d-1787-lite-on-1779-family.md`, `research/specs/030-025b-seed314-new-ttt.md`

## Hypothesis

After testing the `#1787`-lite training-side bundle, the next architectural
question is whether the sparse attention-output gate also transfers onto our
frozen `alpha/beta` family.

This should be tested separately so we do not confound:

- training recipe gains
- with gate-path architecture changes

## Baseline

Use `035d` as the immediate intended predecessor.

Fallback `4×H` reference:

- `026` screen seed `314`: pre-quant `1.06770372`

## Config diff

This is the first sparse-gate test on the `#1779` / `030` family.

Intended changes relative to `035d`:

- sparse-gate code lineage via this branch
- `GATED_ATTN_ENABLED=0`
- `SPARSE_ATTN_GATE_ENABLED=1`
- `SPARSE_ATTN_GATE_INIT_STD=0.0`
- `SPARSE_ATTN_GATE_SCALE=1.0`

Recommended testing order:

1. sparse gate on top of a known-good `035d`-style stack
2. only after that, consider sparse gate in isolation if needed

The older `035c` Polar-NS-only slot is retired in favor of `035d` as the
non-sparse predecessor.

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT for the first test
- `NUM_LOOPS=2`

## Why separate from 035d

Sparse gate is the least certain and most integration-sensitive add in the
public `#1787` bundle:

- changes the model path
- affects training and eval parity
- must be mirrored in all relevant forward paths later if promoted

So it should not be bundled into the first transfer test.

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

## Run protocol

First rung only:

- `035eA`
- seed `314`
- no TTT
- sparse gate on top of a `035d`-style base stack

Execution rule:

- launch from `exp/035e-sparse-gate-on-1779-family`
- use runnable code commit `d205ff5`
- match the intended `035d` base stack exactly
- apply only the sparse-gate code / env diffs
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d205ff5

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NE/NA layout" >&2
  exit 1
fi

mkdir -p /workspace/runs/035e-sparse-gate-on-1779-family/run_a/seed_314
mkdir -p /tmp/torch_inductor_cache_035e_a

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/035e-sparse-gate-on-1779-family/run_a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035e_a \
CASEOPS_ENABLED=1 \
TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0 \
RECUR_ALPHA_ENABLED=1 \
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
  > /workspace/runs/035e-sparse-gate-on-1779-family/run_a/seed_314/train.log 2>&1
```

## What to watch

- pre-quant post-EMA `val_bpb`
- any train/eval path mismatch
- any throughput or artifact-size effect

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log
- note whether sparse gate was active and dense gated-attn was disabled

## Sanity gate

Before accepting the result, execution must verify from `config.json` that the
full `035d` bundle is still present:

- Polar NS code lineage
- `MIN_LR=0.10`
- `FUSED_CE_ENABLED=1`
- `GPTQ_RESERVE_SECONDS=0.5`
- `VAL_LOSS_EVERY=0`

and that the only additional diffs beyond `035d` are:

- sparse-gate code lineage
- `GATED_ATTN_ENABLED=0`
- `SPARSE_ATTN_GATE_ENABLED=1`
- `SPARSE_ATTN_GATE_INIT_STD=0.0`
- `SPARSE_ATTN_GATE_SCALE=1.0`

## Accept criteria

Strong success:

- beats the best non-sparse sibling and clearly justifies keeping sparse gate

Weak success:

- roughly ties but saves enough artifact bytes / path quality to stay alive

Failure:

- worse than the same non-sparse base stack

## Notes

- This should come after `035d`, not before.
- The main point is attribution: first prove the training-side recipe transfers,
  then isolate whether sparse gate adds anything on top.
