# Spec 039 — negative-slope screen on the `#1797` / `038` Smear+LQER base

**Slug:** `neg-slope-screen-on-1797-base`
**Created:** 2026-04-24
**Updated:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/ideas/039-neg-slope-screen-on-1797-base.md`

## Backward bug fix

Original commit `b1f1f8c` had a bug in the fused `LeakyReLU(s)²` Triton
backward: it computed `2·s·x` for negative-side inputs instead of `2·s²·x`.
Each arm was training with an inconsistent forward/backward pair — forward used
slope `s`, backward used the gradient that corresponds to slope `√s`:

| Intended s | Effective backward s |
|---|---|
| 0.3 | √0.3 ≈ 0.548 |
| 0.4 | √0.4 ≈ 0.632 |
| 0.5 | √0.5 ≈ 0.707 |
| 0.6 | √0.6 ≈ 0.775 |

Repinned to `5bbf12f` (fix commit on `exp/039b-loop-band-activation-screen`)
which corrects the backward to `2·s²·x`. All prior 039 results are invalid.

## Hypothesis

On top of the `#1797`-family Smear+LQER stack, the current
`LeakyReLU(0.5)^2` MLP may be slightly too permissive on the negative side.
Reducing the negative slope to `0.4` or `0.3` could modestly improve final BPB,
possibly by lowering quantization damage while still keeping negative-side
gradient flow.

## Baseline

Use the current `038` Smear+LQER line as the execution base.

Reference target from the public `#1797` PR:

- pre-TTT mean: `1.07443`
- post-TTT mean: `1.06157`
- artifact mean: `15,952,695`

Activation baseline:

- `NEGATIVE_SLOPE=0.5`

## Config diff

Relative to the `038` Smear+LQER base:

- expose `NEGATIVE_SLOPE` as a real config/env knob in the MLP path
- thread that value through:
  - eager MLP forward
  - fused Triton MLP path
  - fused Triton backward derivative path
- do not change:
  - tokenizer
  - smear gate
  - LQER asym
  - quant bits / clip policy
  - TTT recipe
  - recurrence / carry / sparse-gate logic

Sweep values:

- `0.3`
- `0.4`
- `0.5` baseline
- `0.6`

Recommended priority order:

1. `0.4`
2. `0.3`
3. `0.6`

Pinned runnable code source:

- branch: `exp/039b-loop-band-activation-screen`
- commit: `5bbf12f`
- script:
  [train_gpt.py](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/039b-loop-band-activation/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py)

## Regime

This is a flat training-only screen, not a promotion.

Run shape:

1. `4×H100`
2. one seed
3. four short runs, one after another
4. no TTT, no full promotion logic

Pinned short-run intent:

- `MAX_WALLCLOCK_SECONDS=180`
- training-only comparison
- same seed for all four runs
- same budget for all four runs

Compare and report:

- steps reached
- train loss trajectory
- validation loss / BPB if the normal training path emits it cheaply
- train time

## Seed policy

Use one fixed seed for the whole screen:

- `42`

Do not fan out to more seeds in this experiment.

## Hardware ladder

1. `4×H100` only
2. no `8×H100` in this spec

## Run protocol

Run these four settings sequentially:

- `NEGATIVE_SLOPE=0.3`
- `NEGATIVE_SLOPE=0.4`
- `NEGATIVE_SLOPE=0.5`
- `NEGATIVE_SLOPE=0.6`

Pinned per-run settings:

- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=180`
- `TTT_ENABLED=0`
- training only

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
```

Suggested execution form:

```bash
for slope in 0.3 0.4 0.5 0.6; do
  env \
    DATA_DIR=/workspace/parameter-golf/data \
    DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin \
    VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin \
    VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin \
    VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8 MLP_MULT=4.0 \
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
    SEED=42 MAX_WALLCLOCK_SECONDS=180 TTT_ENABLED=0 NEGATIVE_SLOPE="$slope" \
    RUN_ID="039-neg-slope-${slope}" \
    torchrun --standalone --nproc_per_node=4 train_gpt.py
done
```

Experiment validity rule:

- the fused and eager MLP paths must use the same configured `NEGATIVE_SLOPE`
- the fused backward must use `2·s²·x` (not `2·s·x`) for negative-side inputs
- must be run on commit `5bbf12f` or later

## Acceptance

Interesting outcome:

- one slope is consistently better than `0.5` on the short training trajectory
- one slope reaches lower validation loss / BPB at the same short wallclock

Practical threshold for continuing:

- one candidate is clearly best on the 3-minute screen and worth a longer
  follow-up

Kill criteria:

- all four slopes look flat / noisy
- any apparent ordering is inconsistent with step count and wallclock

## Open questions

- Is the current fused MLP implementation written in a way that cleanly accepts
  a runtime negative slope, or do we need a temporary compile-time constant?
- Is the short training signal stable enough to justify a longer run?
- If one value wins early, does it survive quantization later or is it just a
  local optimization effect?
