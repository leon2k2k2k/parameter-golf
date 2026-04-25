# Spec 039bN — non-uniform MLP width banding 040C (keep early, widen middle, shrink late)

**Slug:** `mlp-band-activation-screen`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `92886ff`
**Links to:** `research/specs/039bL-floor-then-linear-no-carry-screen.md`

## Background

All LR schedule variants (039bG–039bM) converged to baseline at the end. This
spec attacks a different axis: **non-uniform MLP depth**. Loop layers 3–5
do more work per token — they see the activation twice per loop iteration.

A prior 2×H100 quick screen across three banding patterns (040A/B/C) found:
- 040A (shrink both sides ×3.625/5.0/3.625): throughput hit + underperformed
- 040B (shrink early, keep late ×3.0/5.0/4.0): mild learning regression
- **040C (keep early, shrink late ×4.0/5.0/3.4): stayed on baseline pace — best survivor**

This spec runs 040C on the corrected codebase (leaky backward fixed) stacked on 039bL.

## Hypothesis

Early layers (0-2) do feature extraction and need their full width. Late layers
(6-10) close to readout can spare capacity. Widen the recurrent loop core (3-5)
at the expense of late layers only.

## Budget math

| Group | Layers | mult | hidden | 2×dim×h/layer | subtotal |
|---|---|---|---|---|---|
| early | 0,1,2 | 4.0 | 2048 | 2,097,152 | 6,291,456 |
| middle | 3,4,5 | 5.0 | 2560 | 2,621,440 | 7,864,320 |
| late | 6–10 | 3.4 | 1741 | 1,784,832 | 8,924,160 |
| **total** | | | | | **~23,069,696** |

Width-units: `3×4.0 + 3×5.0 + 5×3.4 = 44.0` — budget-neutral (3.4×512=1740.8 rounds to 1741, +1024 params vs uniform, negligible).

## Config diff

Stacked on 039bL (floor_then_linear + no frozen carry), four new vars:

```bash
LR_SCHEDULE_MODE=floor_then_linear   # from 039bL
LR_REWARM_AT=0.35                    # from 039bL
RECUR_ALPHA_ENABLED=0                # from 039bL
MLP_SCHEDULE_ENABLED=1               # new — enable per-layer width
MLP_EARLY_MULT=4.0                   # new — keep early at baseline
MLP_MIDDLE_MULT=5.0                  # new — widen loop core (layers 3,4,5)
MLP_LATE_MULT=3.4                    # new — shrink late layers 6-10
```

`MLP_MIDDLE_LAYERS=3,4,5` is the existing default — no change needed.

## Implementation notes

Bank allocates `max(hidden_dims)=2560` wide. Each layer slices `[:hd]` in:
- `_bank_weights(i)` — forward pass uses correct slice
- `_init_weights` — zeros don't get orthogonal init
- `GPT.__init__` — bank tensor is allocated at max width

Leaky backward: `2.0 * NEGATIVE_SLOPE * NEGATIVE_SLOPE * pre0` (correct, verified).

## Regime

- `4×H100`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`
- Regions: AP-JP-1 or US-NE-1

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
export RECUR_ALPHA_ENABLED=0
export RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export LR_SCHEDULE_MODE=floor_then_linear LR_REWARM_AT=0.35
export MLP_SCHEDULE_ENABLED=1 MLP_EARLY_MULT=4.0 MLP_MIDDLE_MULT=5.0 MLP_LATE_MULT=3.4
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0
export RUN_ID="039bN-mlp-band-activation"

mkdir -p /workspace/runs/039-neg-slope-screen-on-1797-base/039bN

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/039-neg-slope-screen-on-1797-base/039bN/train.log 2>&1
```

## Compare against

| Arm | pre-quant BPB | Notes |
|---|---|---|
| baseline | 1.06514 | target |
| 039bL (floor_then_linear + no carry) | pending | base for this spec |
| **039bN** (+ MLP banding) | — | this spec |

## Acceptance

- **Win**: pre-quant EMA val_bpb < 1.0641
- **Interesting**: beats 039bL, even if not baseline — banding adds signal
- **Kill**: pre-quant ≥ 1.0660
