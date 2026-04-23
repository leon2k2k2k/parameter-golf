# Spec 036 — `035e` `8×H` promotion with standard `#1779` TTT

**Slug:** `035e-8h-promotion`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/036-035e-8h-promotion`
**Commit:** `8115a0b`
**Links to:** `research/specs/035e-sparse-gate-on-1779-family.md`, `research/specs/030-025b-seed314-new-ttt.md`, `runs/035-series-report.md`

## Hypothesis

`035eA` was the strongest completed `4×H` screen in the current `035` family:

- `035eA` pre-quant `val_bpb = 1.06617649`
- better than `035A` by `-0.00061403`
- better than `035dA` by `-0.00112713`

So the next practical question is no longer whether sparse gate helps on the
screen rung. It is whether the `035e` training stack survives promotion to the
real `8×H` full pipeline with the normal `#1779` / `030` phased LoRA-TTT path.

## Baseline

Primary promotion baseline:

- `030` family `8×H` line with standard phased LoRA-TTT

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

Promotion source:

- `035eA` `4×H` pre-quant: `1.06617649`

## Config diff

Relative to the successful `035eA` `4×H` screen stack:

- promote to `8×H100`
- enable the normal `030` / `#1779` phased LoRA-TTT path
- preserve the successful `035e` sparse-gate training stack exactly
- keep `VAL_LOSS_EVERY=0` on the `8×H` promotion run

Inherited successful `035e` training-side stack:

- `MIN_LR=0.10`
- Polar NS lineage
- `FUSED_CE_ENABLED=1`
- `GPTQ_RESERVE_SECONDS=0.5`
- `VAL_LOSS_EVERY=0`
- sparse gate on, dense gated-attn off

Pinned TTT-side intent:

- `TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`
- normal `030` / `#1779` LoRA-TTT path

Pinned runnable code source:

- branch: `exp/035e-sparse-gate-on-1779-family`
- runnable code commit: `0e13ad0`

## Regime

This is a full `8×H100` promotion run.

Pinned intent:

- same model/training stack as the successful `035eA`
- same frozen recurrent `alpha/beta`
- same sparse gate path
- full quantized eval + phased LoRA-TTT

## Seed policy

Promotion seed is runtime-selectable from the approved shortlist:

- `42`
- `314`
- `2025`
- `777`

Recommended first seed:

- `42`

Execution may choose any one of the approved seeds at launch, but must record
the chosen seed in `notes.md` and `config.json`.

## Hardware ladder

0. optional smoke: `8×H100`, `2` minutes, no TTT, compile/preflight only
1. `8×H100` full pipeline, first promotion seed from the approved shortlist

Optional later:

2. additional seeds if the first run is competitive

## Run protocol

Optional smoke rung:

- `036-smoke`
- `8×H100`
- `MAX_WALLCLOCK_SECONDS=120`
- `TTT_ENABLED=0`
- same training stack otherwise
- discard result; use only for compile/path warmup

First promotion rung:

- `036A`
- `8×H100`
- full quantized eval + phased LoRA-TTT
- preserve the successful `035e` training stack
- seed chosen at launch from the approved shortlist

Execution rule:

- launch from `exp/035e-sparse-gate-on-1779-family`
- use the actually successful runnable code commit `0e13ad0`
- match the successful `035eA` training stack exactly
- only add the standard `030` / `#1779` full-pipeline / TTT settings
- allow execution to choose `SEED` from:
  - `42`
  - `314`
  - `2025`
  - `777`
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned smoke command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 0e13ad0

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NE/NA layout" >&2
  exit 1
fi

: "${SEED_CHOICE:=42}"
case "$SEED_CHOICE" in
  42|314|2025|777) ;;
  *) echo "SEED_CHOICE must be one of: 42, 314, 2025, 777" >&2; exit 1 ;;
esac

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
MAX_WALLCLOCK_SECONDS=120 \
TRAIN_LOG_EVERY=100 \
SEED=$SEED_CHOICE \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/036-035e-8h-promotion/smoke_seed_${SEED_CHOICE}/train.log 2>&1
```

Pinned full-pipeline command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 0e13ad0

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NE/NA layout" >&2
  exit 1
fi

: "${SEED_CHOICE:=42}"
case "$SEED_CHOICE" in
  42|314|2025|777) ;;
  *) echo "SEED_CHOICE must be one of: 42, 314, 2025, 777" >&2; exit 1 ;;
esac

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
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=$SEED_CHOICE \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/036-035e-8h-promotion/seed_${SEED_CHOICE}/train.log 2>&1

kill $NVSMI_PID
```

Smoke-run rule:

- if execution wants compile/path warmup first, use a separate `036-smoke` rung
- only change:
  - `MAX_WALLCLOCK_SECONDS=120`
  - `TTT_ENABLED=0`
- do not compare the smoke result against baselines
- do not treat the smoke run as a quality signal

## What to watch

- post-EMA pre-quant `val_bpb`
- quantized diagnostic `val_bpb`
- final phased-TTT `val_bpb`
- whether sparse gate remains stable and consistent under the full path

## Required artifacts

- training log
- `config.json`
- final pre-quant metrics
- final quantized diagnostic
- final phased-TTT result
- checkpoint and artifact paths needed for postmortem / replay

Smoke rung emits:

- training log
- `config.json`
- nothing from quantized eval / TTT is required

## Acceptance

Primary comparison target:

- `030` seed `777` post-TTT: `1.06428960`

Decision bands:

| post-TTT bpb | verdict | action |
|---|---|---|
| `< 1.0643` | beats current strongest `030` seed | promote immediately to multi-seed |
| `[1.0643, 1.0650]` | competitive with strong `030` seeds | run at least one more approved seed |
| `(1.0650, 1.0660]` | positive but not clearly frontier-leading | compare against `030` control appetite before expanding |
| `> 1.0660` | weak promotion | stop |

## Notes

- The important thing already settled is the base branch/commit:
  `035e` should promote from the actually successful `0e13ad0` line, not the
  earlier stale spec pin.
- `VAL_LOSS_EVERY=0` is intentional for this `8×H` promotion. Do not add
  a step-4000 val checkpoint unless explicitly requested.
- the optional smoke rung is only for compile/preflight warming on the same
  `8×H` stack; it is not part of the quality comparison.
