# Spec 039bM — loop depth curriculum (1→2 loops at 65%)

**Slug:** `loop-depth-curriculum-screen`
**Created:** 2026-04-25
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `0516020`
**Links to:** `research/specs/039bL-floor-then-linear-no-carry-screen.md`

## Background

All LR schedule experiments (039bG/H/I/J/K/L) failed to beat baseline — the
early advantage always erodes because the second half has less LR to spend.

This spec tries a different axis: **recurrence depth curriculum**. Instead of
changing the LR shape, we phase how much recurrence the model uses:

- 0–7min (frac 0–0.35): feedforward only (ENABLE_LOOPING_AT=0.35)
- 7–13min (frac 0.35–0.65): 1 loop iteration (num_loops−1=1)
- 13–20min (frac 0.65–1.0): full 2 loop iterations — the late kick

The upgrade at 13min (the convergence phase) adds a second recurrent pass when
the model is already partially settled, giving it extra refinement capacity
right when LR is declining and the model needs to squeeze out final gains.

This is already implemented in the codebase via `LOOP_DEPTH_UPGRADE_AT`.

## Hypothesis

Activating the second loop iteration late (during warmdown) gives a structural
kick in the convergence phase without the throughput penalty of early activation.
The model has 7min of single-loop adaptation before the upgrade, so the second
pass fires into an already-adapted recurrent stack.

## Config diff

Stacked on 039bL (floor_then_linear + no frozen carry), three changes from baseline:

```bash
LR_SCHEDULE_MODE=floor_then_linear   # from 039bL
LR_REWARM_AT=0.35                    # from 039bL
RECUR_ALPHA_ENABLED=0                # from 039bL
LOOP_DEPTH_UPGRADE_AT=0.65           # new — upgrade 1→2 loops at 13min
```

`ENABLE_LOOPING_AT=0.35` unchanged (loop activates at 7min with 1 iteration).

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
export LOOP_DEPTH_UPGRADE_AT=0.65
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
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0
export RUN_ID="039bM-loop-depth-curriculum"

mkdir -p /workspace/runs/039-neg-slope-screen-on-1797-base/039bM

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/039-neg-slope-screen-on-1797-base/039bM/train.log 2>&1
```

## Compare against

| Arm | pre-quant BPB | Notes |
|---|---|---|
| baseline | 1.06514 | target |
| 039bL (floor_then_linear + no carry) | pending | base for this spec |
| **039bM** (+ depth curriculum) | — | this spec |

## Acceptance

- **Win**: pre-quant EMA val_bpb < 1.0641
- **Interesting**: beats 039bL, even if not baseline — depth curriculum adds signal
- **Kill**: pre-quant ≥ 1.0660
