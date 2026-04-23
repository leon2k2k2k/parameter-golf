# Spec 036 — sparse-gate `8×H` promotion with updated rounded `alpha/beta`

**Slug:** `sparse-updated-alpha-beta-8h-promotion`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/036-035e-8h-promotion`
**Commit:** `71ad359`
**Links to:** `research/specs/035e-sparse-gate-on-1779-family.md`, `research/specs/035h-learnable-alpha-beta-on-sparse-gate-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `runs/035-series-report.md`

## Hypothesis

`035eA` was the strongest completed `4×H` screen in the current `035` family.
The next promotion is therefore the sparse-gate stack at `8×H`, using the
updated recurrent carry learned on the sparse learnable-alpha/beta run `035h`
and then frozen back into the model.

## Baseline

Primary promotion baselines:

- `030` family `8×H` line with standard phased LoRA-TTT
- `035eA` `4×H` sparse-gate screen at `1.06617649`

Reference points:

- `030` seed `314`:
  - pre-quant `1.06821629`
  - post-TTT `1.06471941`
- `030` seed `2025`:
  - pre-quant `1.06821738`
  - post-TTT `1.06438348`
- `030` seed `777`:
  - pre-quant `1.06798687`
  - post-TTT `1.06428960`

Updated rounded recurrent carry to promote:

- source: final `035h` learned `alpha/beta`
- rounding rule: **2 decimal places**, not 2 significant digits
- exact source values:
  - `beta = [1.5610, 1.8531, 2.1320]`
  - `alpha = [[0.2314, 0.0388, 0.0347], [0.1260, -0.3438, 0.0145], [0.0557, 0.1934, -0.0172]]`
- frozen rounded values used in `036`:
  - `beta = [1.56, 1.85, 2.13]`
  - `alpha = [[0.23, 0.04, 0.03], [0.13, -0.34, 0.01], [0.06, 0.19, -0.02]]`

## Config diff

Relative to the successful `035eA` `4×H` screen stack:

- promote to `8×H100`
- enable the normal `030` / `#1779` phased LoRA-TTT path
- preserve the successful `035e` sparse-gate training stack
- replace the old baked `025b` recurrent carry with the updated rounded frozen
  carry from the later sparse learnable-alpha/beta line `035h`
- keep `VAL_LOSS_EVERY=0` on the `8×H` promotion run

Inherited successful `035e` training-side stack:

- `MIN_LR=0.10`
- Polar NS lineage
- `FUSED_CE_ENABLED=1`
- `GPTQ_RESERVE_SECONDS=0.5`
- `VAL_LOSS_EVERY=0`
- sparse gate on, dense gated-attn off
- recurrent carry frozen to the rounded `035h` values above

Pinned TTT-side intent:

- `TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`
- normal `030` / `#1779` LoRA-TTT path

Pinned runnable code source:

- shell/spec branch: `exp/036-035e-8h-promotion`
- runnable code branch: `exp/036-sparse-updated-alpha-beta`
- runnable code commit: `1d12cb6`

## Regime

This is a full `8×H100` promotion run.

Pinned intent:

- same model/training stack as the successful `035eA`
- sparse gate path from `035e`
- updated rounded frozen recurrent `alpha/beta` from `035h`
- full quantized eval + phased LoRA-TTT

## Seed policy

Promotion seed is runtime-selectable from the approved shortlist:

- `1`
- `777`
- `2025`

Recommended first seed:

- `1`

Execution may choose any one of the approved seeds at launch, but must record
the chosen seed in `notes.md` and `config.json`.

## Hardware ladder

0. optional smoke: `8×H100`, `600s`, no TTT, compile/preflight only
1. `8×H100` full pipeline, `600s`, first promotion seed from the approved shortlist

Optional later:

2. additional seeds if the first run is competitive

## Run protocol

Optional smoke rung:

- `036-smoke`
- `8×H100`
- `MAX_WALLCLOCK_SECONDS=600`
- `TTT_ENABLED=0`
- same training stack otherwise
- discard result; use only for compile/path warmup

First promotion rung:

- `036A`
- `8×H100`
- full quantized eval + phased LoRA-TTT
- preserve the successful `035e` sparse-gate stack
- use the updated rounded frozen recurrent carry from `035h`
- seed chosen at launch from the approved shortlist

TTT bugfix note:

- `036A` seed `1` showed healthy pre-quant and quantized eval but catastrophic
  post-TTT (`1.8089`)
- root cause was a code bug: sparse gate existed in normal eval but was missing
  from the two LoRA-TTT forward paths
- runnable commit `1d12cb6` fixes sparse gate in both `_block_with_lora(...)`
  and `_parallel_block_with_lora(...)`

Execution rule:

- launch from `exp/036-sparse-updated-alpha-beta`
- use runnable code commit `1d12cb6`
- keep the successful `035eA` sparse-gate stack otherwise
- add the standard `030` / `#1779` full-pipeline / TTT settings
- allow execution to choose `SEED` from:
  - `1`
  - `777`
  - `2025`
- require `config.json`
- if the produced config drifts on anything except the intended updated frozen
  recurrent carry, the rung is invalid

Pinned smoke/full commands:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 74a060d

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NE/NA layout" >&2
  exit 1
fi

: "${SEED_CHOICE:=1}"
case "$SEED_CHOICE" in
  1|777|2025) ;;
  *) echo "SEED_CHOICE must be one of: 1, 777, 2025" >&2; exit 1 ;;
esac
```

Smoke:

```bash
mkdir -p /workspace/runs/036-035e-8h-promotion/smoke_seed_${SEED_CHOICE}
mkdir -p /tmp/torch_inductor_cache_036_smoke

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/036-035e-8h-promotion/smoke_seed_${SEED_CHOICE} \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_036_smoke \
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
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=100 \
SEED=$SEED_CHOICE \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/036-035e-8h-promotion/smoke_seed_${SEED_CHOICE}/train.log 2>&1
```

Full pipeline:

```bash
mkdir -p /workspace/runs/036-035e-8h-promotion/seed_${SEED_CHOICE}
mkdir -p /tmp/torch_inductor_cache_036_8h

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/036-035e-8h-promotion/seed_${SEED_CHOICE}/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/036-035e-8h-promotion/seed_${SEED_CHOICE} \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_036_8h \
CASEOPS_ENABLED=1 \
TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
MUON_BACKEND_STEPS=5 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
VAL_LOSS_EVERY=0 \
FUSED_CE_ENABLED=1 \
MIN_LR=0.10 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=100 \
SEED=$SEED_CHOICE \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/036-035e-8h-promotion/seed_${SEED_CHOICE}/train.log 2>&1

kill $NVSMI_PID
wait $NVSMI_PID 2>/dev/null || true
```

## Acceptance

This run is interesting if it is competitive with the strong `030` post-TTT
range and clearly validates the sparse-gate plus updated-carry promotion path.

Primary target:

- final post-TTT `val_bpb` competitive with the better `030` seeds

Secondary target:

- no sign that the updated rounded carry breaks the successful sparse-gate line

## Why this is now ready

The open parameter question is resolved:

- source run is `035h`
- rounding is 2 decimal places
- the runnable code branch now bakes in the resulting frozen values
