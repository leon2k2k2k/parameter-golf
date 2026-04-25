# Spec 040 — loop-layer MLP reallocation screen

**Slug:** `loop-layer-mlp-reallocation-screen`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/040-loop-layer-mlp-reallocation-screen`
**Commit:** `983559f`
**Links to:** `research/ideas/040-loop-layer-mlp-reallocation.md`, `research/specs/039-neg-slope-screen-on-1797-base.md`

## Hypothesis

The looped middle physical layers in the current `038/039` family deserve more
FFN capacity than the outer layers. Reallocating MLP width toward the middle
loop band should outperform the current uniform `MLP_MULT=4.0` allocation at
roughly fixed total MLP parameter count.

## Baseline

Use the corrected `039` code line as the starting base.

Pinned current runnable base:

- branch: `exp/039-neg-slope-screen-on-038-fullfloat-base`
- commit: `b1f1f8c`

Assume the activation screen either:

- is still pending, in which case use `NEGATIVE_SLOPE=0.5`
- or has produced a winner, in which case execution may substitute the winning
  slope if research explicitly updates this spec before launch

Uniform width baseline:

- all `11` physical layers at `MLP_MULT=4.0`

## Config diff

Keep the whole `038/039` stack fixed and change only the per-layer MLP width
schedule.

Pinned implementation API for this spec:

- `MLP_SCHEDULE_ENABLED=1`
- `MLP_EARLY_MULT=<float>`
- `MLP_MIDDLE_MULT=<float>`
- `MLP_LATE_MULT=<float>`
- `MLP_MIDDLE_LAYERS=3,4,5`
- `TRAINING_ONLY_SCREEN=1`

Interpretation:

- early layers are all physical layers before the middle list
- middle layers are exactly `MLP_MIDDLE_LAYERS`
- late layers are all physical layers after the middle list

Pinned runnable code source:

- branch: `exp/040-loop-layer-mlp-reallocation-screen`
- commit: `983559f`
- script:
  [train_gpt.py](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/038-fullfloat-smear-lqer-asym/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py)

Looped middle physical layers are:

- `3,4,5`

Three schedule arms:

### 040A — shrink both sides evenly

- early `0,1,2` -> `3.625x`
- middle `3,4,5` -> `5.0x`
- late `6,7,8,9,10` -> `3.625x`

### 040B — shrink early, keep late

- early `0,1,2` -> `3.0x`
- middle `3,4,5` -> `5.0x`
- late `6,7,8,9,10` -> `4.0x`

### 040C — shrink late, keep early

- early `0,1,2` -> `4.0x`
- middle `3,4,5` -> `5.0x`
- late `6,7,8,9,10` -> `3.4x`

All three schedules are matched to the baseline total of `44.0` width-units.

Implementation note:

- code only needs to support per-layer or per-band MLP width allocation for the
  training path in this spec
- quantization / serialization / deserialize support is explicitly out of scope
  for the first `040` screen
- if exact fractional widths are awkward, choose the nearest implementation
  representation but preserve the schedule intent and total-width matching as
  closely as practical

## Regime

This is an explicitly training-only screen.

Pinned short-run intent:

- `4×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=600`
- `TTT_ENABLED=0`
- `NEGATIVE_SLOPE=0.5` unless `039` is explicitly promoted first

Compare:

- steps reached
- train loss trajectory
- validation loss / BPB if emitted cheaply
- train time

Out of scope for this spec:

- GPTQ / quantized artifact generation
- deserialize / rebank compatibility for variable-width MLPs
- TTT

## Seed policy

Use one seed only:

- `42`

## Hardware ladder

1. `4×H100` only
2. no `8×H100` in this spec
3. no quantized eval in this spec

## Run protocol

Run four training-only short jobs:

1. uniform baseline `MLP_MULT=4.0`
2. `040A` shrink both evenly
3. `040B` shrink early, keep late
4. `040C` shrink late, keep early

Same seed, same wallclock, same env otherwise.

Execution rule:

- do not attempt to serialize or evaluate the quantized model in this spec
- if the current code path automatically enters GPTQ / serialize / TTT after
  training, research must add or use a training-only stop path before launch

Resolved base env block:

```bash
DATA_DIR=/workspace/parameter-golf/data
DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin
VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin
VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin
VOCAB_SIZE=8192
NUM_LAYERS=11
XSA_LAST_N=11
MODEL_DIM=512
NUM_KV_HEADS=4
NUM_HEADS=8
MLP_MULT=4.0
NEGATIVE_SLOPE=0.5
TIE_EMBEDDINGS=1
LOGIT_SOFTCAP=30
ROPE_BASE=10000
ROPE_DIMS=16
ROPE_TRAIN_SEQ_LEN=2048
ROPE_YARN=0
LN_SCALE=1
QK_GAIN_INIT=5.0
NUM_LOOPS=2
LOOP_START=3
LOOP_END=5
ENABLE_LOOPING_AT=0.35
PARALLEL_START_LAYER=8
PARALLEL_FINAL_LANE=mean
MIN_LR=0.1
EMBED_LR=0.6
TIED_EMBED_LR=0.03
TIED_EMBED_INIT_STD=0.005
MATRIX_LR=0.026
SCALAR_LR=0.02
MUON_MOMENTUM=0.97
MUON_BACKEND_STEPS=5
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_ROW_NORMALIZE=1
BETA1=0.9
BETA2=0.95
ADAM_EPS=1e-8
GRAD_CLIP_NORM=0.3
ADAM_WD=0.02
MUON_WD=0.095
EMBED_WD=0.085
EMA_DECAY=0.9965
TRAIN_BATCH_TOKENS=786432
TRAIN_SEQ_LEN=2048
TRAIN_LOG_EVERY=100
ITERATIONS=20000
WARMDOWN_FRAC=0.75
WARMUP_STEPS=20
VAL_BATCH_TOKENS=524288
EVAL_SEQ_LEN=2048
EVAL_STRIDE=64
VAL_LOSS_EVERY=0
CASEOPS_ENABLED=1
COMPRESSOR=brotli
MATRIX_BITS=6
MATRIX_CLIP_SIGMAS=12.85
ATTN_CLIP_SIGMAS=13.0
MLP_CLIP_SIGMAS=12.0
EMBED_BITS=7
EMBED_CLIP_SIGMAS=15.0
GPTQ_CALIBRATION_BATCHES=16
GPTQ_RESERVE_SECONDS=0.5
SKIP_GATES_ENABLED=1
SPARSE_ATTN_GATE_ENABLED=1
SPARSE_ATTN_GATE_INIT_STD=0.0
SPARSE_ATTN_GATE_SCALE=1.0
GATED_ATTN_ENABLED=0
GATED_ATTN_INIT_STD=0.005
GATED_ATTN_QUANT_GATE=1
ATTN_OUT_GATE_ENABLED=0
ATTN_OUT_GATE_SRC=proj
GATE_WINDOW=12
RECUR_ALPHA_ENABLED=1
RECUR_DIAG_P2P_COS=0
SMEAR_GATE_ENABLED=1
LQER_ENABLED=1
LQER_RANK=4
LQER_TOP_K=3
LQER_FACTOR_BITS=4
LQER_ASYM_ENABLED=1
LQER_ASYM_GROUP=64
SPINQUANT_ENABLED=0
SPINQUANT_SEED=42
SPINQUANT_SITES=attn_in,attn_proj_in,mlp_in,mlp_proj_in
SEED=42
MAX_WALLCLOCK_SECONDS=600
TTT_ENABLED=0
MLP_SCHEDULE_ENABLED=1
MLP_MIDDLE_LAYERS=3,4,5
TRAINING_ONLY_SCREEN=1
```

Canonical launch block:

```bash
declare -A EARLY_MULT=(
  [baseline]=4.0
  [040A]=3.625
  [040B]=3.0
  [040C]=4.0
)

declare -A MIDDLE_MULT=(
  [baseline]=4.0
  [040A]=5.0
  [040B]=5.0
  [040C]=5.0
)

declare -A LATE_MULT=(
  [baseline]=4.0
  [040A]=3.625
  [040B]=4.0
  [040C]=3.4
)

for arm in baseline 040A 040B 040C; do
  env \
    DATA_DIR=/workspace/parameter-golf/data \
    DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin \
    VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin \
    VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin \
    VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8 MLP_MULT=4.0 NEGATIVE_SLOPE=0.5 \
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16 ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0 \
    NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean \
    MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005 MATRIX_LR=0.026 SCALAR_LR=0.02 \
    MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1 \
    BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085 EMA_DECAY=0.9965 \
    TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100 ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20 \
    VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0 \
    CASEOPS_ENABLED=1 COMPRESSOR=brotli MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=0.5 \
    SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0 \
    GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12 \
    RECUR_ALPHA_ENABLED=1 RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1 \
    LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
    SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES=attn_in,attn_proj_in,mlp_in,mlp_proj_in \
    SEED=42 MAX_WALLCLOCK_SECONDS=600 TTT_ENABLED=0 \
    MLP_SCHEDULE_ENABLED=1 MLP_MIDDLE_LAYERS=3,4,5 TRAINING_ONLY_SCREEN=1 \
    MLP_EARLY_MULT="${EARLY_MULT[$arm]}" \
    MLP_MIDDLE_MULT="${MIDDLE_MULT[$arm]}" \
    MLP_LATE_MULT="${LATE_MULT[$arm]}" \
    RUN_ID="040-${arm}" \
    torchrun --standalone --nproc_per_node=4 train_gpt.py
done
```

## Acceptance

Interesting outcome:

- any non-uniform schedule clearly beats the uniform baseline on the short
  training screen

Most interesting outcome:

- `040B` wins, supporting the story that looped middle layers deserve extra
  FFN budget and that early layers are the cheapest place to give it up

Kill criteria:

- all three reallocations are flat or worse than uniform
- results are dominated by implementation artifacts rather than real learning
  signal

## Open questions

- What is the cleanest code representation for per-band MLP widths in the
  current banked MLP implementation?
- Should the first screen use the activation winner from `039`, or keep
  `NEGATIVE_SLOPE=0.5` to reduce moving parts?
- If one schedule wins, does it survive quantization and TTT, or is the gain
  only visible in short training?
