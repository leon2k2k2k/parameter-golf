# Spec 039bE — early loop activation + LR floor screen

**Slug:** `early-loop-lr-floor-screen`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/specs/039-neg-slope-screen-on-1797-base.md`

## Hypothesis

Two changes stacked on the 039 baseline:

1. `ENABLE_LOOPING_AT=0.175` — activates looping at 17.5% of wallclock
   (~3.5 min into a 20-min run) instead of 35% (~7 min). Already screened in
   039bD and shown to cost throughput with no BPB payoff **when combined with
   penalized_tanh**. This arm isolates whether the loop-timing benefit is real
   on the cleaner leaky_relu_square base.
2. `LR_SCHEDULE_MODE=first_half_default_then_floor` — cosine decay compressed
   into the first 50% of wallclock, then held flat at MIN_LR for the remaining
   50%. Prior art (eval 042C) showed +0.00015 on the penalized_tanh 039bA base
   (within seed noise). Testing on leaky_relu_square may show a cleaner signal.

No penalized_tanh. This is a pure loop-timing + LR-schedule isolation on the
best activation from the 039 screen.

## Baseline

Same 1797-family Smear+LQER stack as the 039 screen baseline:

- branch: `exp/039b-loop-band-activation-screen`
- commit: `5bbf12f`
- reference result (4×H100, ENABLE_LOOPING_AT=0.35, default cosine): pre-quant EMA val_bpb=1.06514, quant val_bpb=1.07410

## Config diff

Relative to the 039 baseline arm:

- change: `ENABLE_LOOPING_AT=0.175` (from `0.35`)
- add: `LR_SCHEDULE_MODE=first_half_default_then_floor`
- keep: `MLP_OUTER_ACTIVATION=leaky_relu_square`, `NEGATIVE_SLOPE=0.5`
- no middle activation changes (no penalized_tanh)

## Regime

- `4×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=1200`
- `TTT_ENABLED=0`
- training-only screen

## Resolved env block

Same as 039 baseline, with two changes:

```bash
ENABLE_LOOPING_AT=0.175
LR_SCHEDULE_MODE=first_half_default_then_floor
```

Full env:

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
ENABLE_LOOPING_AT=0.175
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
MLP_OUTER_ACTIVATION=leaky_relu_square
NEGATIVE_SLOPE=0.5
LR_SCHEDULE_MODE=first_half_default_then_floor
SEED=42
MAX_WALLCLOCK_SECONDS=1200
TTT_ENABLED=0
```

## Suggested launch form

```bash
env \
  ... (full env block above) ... \
  ENABLE_LOOPING_AT=0.175 \
  LR_SCHEDULE_MODE=first_half_default_then_floor \
  MLP_OUTER_ACTIVATION=leaky_relu_square \
  NEGATIVE_SLOPE=0.5 \
  RUN_ID="039bE-early-loop-lr-floor" \
  torchrun --standalone --nproc_per_node=4 train_gpt.py \
  >> /workspace/runs/039-neg-slope-screen-on-1797-base/039bE/train.log 2>&1
```

## Compare against

- 039 baseline: `ENABLE_LOOPING_AT=0.35`, default cosine, pre-quant EMA val_bpb=1.06514, quant val_bpb=1.07410
- 039bD: `ENABLE_LOOPING_AT=0.175` + penalized_tanh, pre-quant EMA val_bpb=1.07520, quant val_bpb=1.08441 (worse — penalized_tanh hurt)
- 042C (prior art): `first_half_default_then_floor` on 039bA base, delta −0.00015 vs 042A (within noise)
- Question: does leaky_relu_square + early loop + LR floor show a cleaner combined signal than either change alone?

## Acceptance

Interesting if pre-quant EMA val_bpb < 1.0641 (baseline) at matched steps.
Kill if no improvement vs baseline after 15 min of training.
