# Spec 041B — 1797-style Smear+LQER plus loop-band penalized-tanh `8×H`

**Slug:** `1797-plus-penalized-tanh-8h`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/041b-1797-penalized-tanh`
**Commit:** `26eb119`

## Hypothesis

The strong `039bA` loop-band `penalized_tanh` signal should be tested on a
`#1797`-style base:

- `SmearGate` on
- `LQER-asym` on
- **no frozen carry**

In this checkout, the faithful way to do that is to use the same runnable
Smear+LQER code line as `039b`, but explicitly disable the frozen `recur_alpha`
carry by env.

## Runnable code source

- branch: `exp/041b-1797-penalized-tanh`
- commit: `26eb119`
- script:
  [train_gpt.py](/home/claude-user/ai-workspace/projects/parameter-golf/worktrees/041b-1797-penalized-tanh/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py)

## Config diff

Relative to the `041A` / `039bA` full-run setup:

- keep:
  - `SMEAR_GATE_ENABLED=1`
  - `LQER_ENABLED=1`
  - `LQER_RANK=4`
  - `LQER_TOP_K=3`
  - `LQER_FACTOR_BITS=4`
  - `LQER_ASYM_ENABLED=1`
  - `LQER_ASYM_GROUP=64`
  - `MLP_OUTER_ACTIVATION=leaky_relu_square`
  - `MLP_MIDDLE_ACTIVATION=penalized_tanh`
  - `MLP_MIDDLE_LAYERS=3,4,5`
- change:
  - `RECUR_ALPHA_ENABLED=0`

So `041B` is the no-frozen-carry sibling of `041A`, not the carry line.

## Regime

- `8×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=600`
- full quantization / deserialize / TTT path enabled
- `TRAIN_LOG_EVERY=100`

## Canonical env block

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
MLP_OUTER_ACTIVATION=leaky_relu_square
MLP_MIDDLE_ACTIVATION=penalized_tanh
MLP_MIDDLE_NEGATIVE_SLOPE=0.5
MLP_MIDDLE_LAYERS=3,4,5
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
RECUR_ALPHA_ENABLED=0
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
TTT_ENABLED=1
RUN_ID=041B-1797-penalized-tanh
```

## Canonical launch

```bash
cd /workspace/parameter-golf
git fetch fork
git checkout exp/041b-1797-penalized-tanh
git reset --hard fork/exp/041b-1797-penalized-tanh

env \
  DATA_DIR=/workspace/parameter-golf/data \
  DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  TRAIN_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin \
  VAL_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin \
  VAL_BYTES_FILES=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin \
  VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8 \
  MLP_MULT=4.0 NEGATIVE_SLOPE=0.5 \
  MLP_OUTER_ACTIVATION=leaky_relu_square \
  MLP_MIDDLE_ACTIVATION=penalized_tanh \
  MLP_MIDDLE_NEGATIVE_SLOPE=0.5 \
  MLP_MIDDLE_LAYERS=3,4,5 \
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
  RECUR_ALPHA_ENABLED=0 RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES=attn_in,attn_proj_in,mlp_in,mlp_proj_in \
  SEED=42 MAX_WALLCLOCK_SECONDS=600 TTT_ENABLED=1 \
  RUN_ID=041B-1797-penalized-tanh \
  torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py
```
