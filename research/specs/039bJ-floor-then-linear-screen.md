# Spec 039bJ — floor_then_linear LR schedule screen

**Slug:** `floor-then-linear-screen`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `0516020`
**Links to:** `research/specs/039bI-floor-then-wsd-loop35-screen.md`, `research/specs/039bH-floor-then-wsd-screen.md`

## Background

039bH/I tested `floor_then_wsd`: fast LR descent in the first 35% of wallclock,
then a jump back up to the standard WSD curve at the switch point. The jump caused
a loss spike in 039bI at step 3000 (+0.006 vs baseline), because loop activation
and the LR jump happen simultaneously at 7min — two destabilizing events at once.

This spec replaces the jump with a smooth linear continuation: at the switch point
(frac=0.35, 7min), instead of jumping to 0.867, the LR simply continues declining
linearly from 0.400 down to MIN_LR=0.1 at the end of the run. No discontinuity,
no shock to the optimizer.

## Hypothesis

The fast first-half descent gives useful early signal. A gentle linear second half
(0.400 → 0.100 over 13min) lets the model converge without the spike. Should
outperform floor_then_wsd (039bH/I) and possibly beat the baseline.

## Schedule: `floor_then_linear`

```
lr_mul(frac):
  rf = frac / 0.5
  if frac < LR_REWARM_AT:          # first_half_floor phase
      if rf >= 1 - WARMDOWN_FRAC: return max((1-rf)/WARMDOWN_FRAC, MIN_LR)
      return 1.0
  # linear from value at switch to MIN_LR at end
  lr_at_switch = max((1 - LR_REWARM_AT/0.5) / WARMDOWN_FRAC, MIN_LR)  # = 0.400
  t = (frac - LR_REWARM_AT) / (1 - LR_REWARM_AT)
  return max(lr_at_switch + (MIN_LR - lr_at_switch) * t, MIN_LR)
```

Key values (LR_REWARM_AT=0.35, WARMDOWN_FRAC=0.75, MIN_LR=0.1):

| Time | frac | floor_then_linear | baseline WSD |
|---|---|---|---|
| 2.5min | 0.125 | 1.000 (warmdown starts) | 1.000 |
| 7min | 0.350 | 0.400 (switch, no jump) | 0.867 |
| 10min | 0.500 | 0.331 | 0.667 |
| 15min | 0.750 | 0.215 | 0.333 |
| 20min | 1.000 | 0.100 | 0.100 |

Implemented in commit `0516020`.

## Baseline

- branch: `exp/039b-loop-band-activation-screen`
- commit: `0516020`
- reference: pre-quant EMA val_bpb=**1.06514**, quant=**1.07410**

## Config diff

Relative to the clean 039 baseline — two changes only:

```bash
LR_SCHEDULE_MODE=floor_then_linear   # new
LR_REWARM_AT=0.35                    # switch point at 7min
# ENABLE_LOOPING_AT stays at 0.35 (baseline default, loop@7min)
```

## Regime

- `4×H100`
- `SEED=42`
- `MAX_WALLCLOCK_SECONDS=1200`
- `TTT_ENABLED=0`
- training-only screen

## Resolved env block

```bash
DATA_DIR=/workspace/parameter-golf/data
DATASETS_DIR=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
VOCAB_SIZE=8192
NUM_LAYERS=11
XSA_LAST_N=11
MODEL_DIM=512
NUM_KV_HEADS=4
NUM_HEADS=8
MLP_MULT=4
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
MLP_OUTER_ACTIVATION=leaky_relu_square
NEGATIVE_SLOPE=0.5
LR_SCHEDULE_MODE=floor_then_linear
LR_REWARM_AT=0.35
SEED=42
MAX_WALLCLOCK_SECONDS=1200
TTT_ENABLED=0
```

## Launch form

```bash
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=0.5
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=1 RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export LR_SCHEDULE_MODE=floor_then_linear LR_REWARM_AT=0.35
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0
export RUN_ID="039bJ-floor-then-linear"

mkdir -p /workspace/runs/039-neg-slope-screen-on-1797-base/039bJ

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/039-neg-slope-screen-on-1797-base/039bJ/train.log 2>&1
```

## Compare against

| Arm | pre-quant BPB | quant BPB | Notes |
|---|---|---|---|
| baseline (loop@0.35, std WSD) | 1.06514 | 1.07410 | target to beat |
| 039bG (loop@0.35, first_half_floor) | 1.06800 | 1.07897 | floor hurts convergence |
| 039bH (loop@0.175, floor_then_wsd) | pending | pending | misaligned re-warm |
| 039bI (loop@0.35, floor_then_wsd) | pending | pending | spike at 7min |
| **039bJ** (loop@0.35, floor_then_linear) | — | — | this spec — no spike |

## Acceptance

- **Win**: pre-quant EMA val_bpb < 1.0641 AND quant BPB ≤ 1.0750
- **Interesting**: pre-quant beats baseline even if quant is flat
- **Kill**: pre-quant ≥ 1.0660

## Open questions

- Does the lower LR in the second half (0.400 → 0.100 vs baseline 0.867 → 0.100)
  hurt final convergence more than the fast first half helps?
- If 039bJ wins, worth trying LR_REWARM_AT=0.25 to get more of the fast phase
  before the linear second half.
