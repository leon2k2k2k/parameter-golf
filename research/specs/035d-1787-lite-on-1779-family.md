# Spec 035d — `#1787`-lite bundle on the `#1779` / `030` alpha/beta family

**Slug:** `1787-lite-on-1779-family`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/035d-1787-lite-on-1779-family`
**Commit:** `2252a83`
**Links to:** `research/ideas/035d-1787-lite-on-1779-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `research/specs/035-min-lr-on-030-family.md`, `research/specs/035b-plateau-on-030-family.md`, `research/ideas/1779-next-adds-ranked.md`

## Hypothesis

The strongest transferable pieces of public `#1787` are not the sparse gate,
but the training-stack refinements:

- `MIN_LR`
- Polar NS
- fused CE
- budget polish

These should be tested together on top of our stronger frozen `alpha/beta`
family (`#1779` / `030`) before spending time on the more invasive sparse-gate
change.

This spec replaces the older `035c` Polar-NS-only slot as the preferred
“next bundle” experiment in the `035` family.

## Baseline

Use the intended `030` `4×H` screen stack as the exact base line.

Primary `4×H` reference:

- `026` screen seed `314`: pre-quant `1.06770372`

Internal family references:

- `035`: same stack + `MIN_LR=0.10`
- `035b`: same stack + plateau schedule

## Config diff

Relative to the intended `030` `4×H` screen stack, the intended bundle is:

- `MIN_LR=0.10`
- Polar NS Newton-Schulz tuples
- fused CE training kernel
- budget polish:
  - `GPTQ_RESERVE_SECONDS=0.5`
  - `VAL_LOSS_EVERY=0`

No sparse gate in this spec.

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT for the first test
- `NUM_LOOPS=2`

## Why this before sparse gate

This bundle is mostly training-side and relatively easy to port:

- schedule (`MIN_LR`)
- optimizer refinement (Polar NS)
- training efficiency (fused CE)
- budget recovery (reserve / val-loss polish)

Sparse gate is the architecture-ish change and should be isolated later.

## Hardware ladder

1. `4×H100` screen, seed `314`, pre-quant only

If clearly positive, then decide whether to:

- promote to a full `8×H` result
- or test the sparse gate as the next architectural add

## Run protocol

First rung only:

- `035dA`
- seed `314`
- no TTT
- full `#1787`-lite bundle

Execution rule:

- launch from `exp/035d-1787-lite-on-1779-family`
- use runnable code commit `2252a83`
- match the intended `030` `4×H` screen stack exactly
- apply only the bundle diffs listed above
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 2252a83

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NE/NA layout" >&2
  exit 1
fi

mkdir -p /workspace/runs/035d-1787-lite-on-1779-family/run_a/seed_314
mkdir -p /tmp/torch_inductor_cache_035d_a

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/035d-1787-lite-on-1779-family/run_a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035d_a \
CASEOPS_ENABLED=1 \
TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
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
  > /workspace/runs/035d-1787-lite-on-1779-family/run_a/seed_314/train.log 2>&1
```

## What to watch

- pre-quant post-EMA `val_bpb`
- any train instability
- any throughput improvement from fused CE / budget polish
- whether the bundle cleanly beats the simpler `035`

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log
- note whether fused CE stayed enabled in the produced config

## Sanity gate

Before accepting the result, execution must verify from `config.json` that the
only intentional diffs from the intended `030` `4×H` screen stack are:

- Polar NS code lineage
- `MIN_LR=0.10`
- `FUSED_CE_ENABLED=1`
- `GPTQ_RESERVE_SECONDS=0.5`
- `VAL_LOSS_EVERY=0`

Data-root rule:

- if the CaseOps tokenizer exists under `/workspace/data/...`, use
  `DATA_DIR=/workspace`
- if it exists under `/workspace/parameter-golf/data/...`, use
  `DATA_DIR=/workspace/parameter-golf/data`
- if neither layout exists, abort

## Accept criteria

Strong success:

- pre-quant clearly beats `1.06770372`
- and is meaningfully better than `035`

Weak success:

- directionally positive enough to justify a full `8×H` follow-up

Failure:

- flat or worse than the `026` `4×H` reference

## Notes

- This is the preferred “move fast” test before sparse gate.
- It intentionally gives up attribution across the training-side adds in
  exchange for a faster answer on whether the public `#1787` bundle transfers
  onto our stronger frozen `alpha/beta` family.
