# Spec 042C — slope anneal 0.707→0.0 at warmdown

**Slug:** `042C-slope-anneal-0707-to-0-screen`
**Created:** 2026-04-26
**Status:** READY
**Branch:** `exp/042-slope-anneal-screen`
**Commit:** `4b29c64`
**Links to:** `research/ideas/slope-annealing-sqrt-to-half.md`

## Hypothesis

Same mechanism as 042A but anneals all the way to `NEGATIVE_SLOPE=0.0` (pure squared ReLU)
instead of 0.5. Two effects compound:

1. **Phase-1 benefit (same as 042A):** Running at s=0.707 for the first 75% gives more
   gradient flow through negative activations, especially around loop activation at frac=0.35.

2. **Phase-2 sparsity bonus:** Pure `relu_square` (s=0) produces ~80-95% natural activation
   sparsity in the FFN intermediate. NVIDIA H100 has native 2:4 structured sparse tensor
   core support — the sparse activations can accelerate FFN GEMMs ~1.3-1.5×, squeezing
   more effective steps into the 1200s wallclock cap.

Unlike 042A (step switch), the slope decays continuously — linearly from 0.7071 to 0.0
over the first 50% of warmdown (frac 0.25→0.625), then holds at 0.0 for the remainder.
This gives the model half of warmdown to fully adapt to pure relu_square before the
final convergence push. `NEGATIVE_SLOPE` is now a runtime Triton arg (not constexpr),
so there is no recompile at any point.

## Config diff

Two env var changes from the 039b baseline:

```bash
NEGATIVE_SLOPE=0.7071
SLOPE_WARMDOWN=0.0
```

All loop/architecture settings identical to baseline (layers 3-5, NL=2, frac=0.35).

## Regime

- `4×H100`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=0`
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
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=1000
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=1
export RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.7071 SLOPE_WARMDOWN=0.0
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=0 VAL_LOSS_EVERY=1000
export RUN_ID="042C-slope-anneal-0707-to-0"

pip install brotli --break-system-packages -q

mkdir -p /workspace/runs/042C-slope-anneal-0707-to-0-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/042C-slope-anneal-0707-to-0-screen/train.log 2>&1
```

## What to watch

- **Slope switch:** `slope_anneal: 0.7071→0.0000 step:~1650 frac:0.250` in log
- **Post-switch throughput:** expect slight increase in tok/s if H100 sparse path engages
- **Post-switch val trajectory:** does it destabilize (spike up) or continue descent?
- **Final EMA val_bpb:** vs baseline 1.06514 and vs 042A result

## Acceptance

Baseline (039b): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641**
- **Noise zone**: 1.0641–1.0670 — compare to 042A result
- **Kill**: pre-quant EMA val_bpb ≥ **1.0670**, or val spikes at slope switch

Post-quant val_bpb also recorded (no TTT).

## Prediction

More aggressive than 042A. The sparsity benefit could add ~50-200 extra effective steps
via throughput gain. Risk is warmdown instability from the full zero-out of negative slopes.
Optimistic: ~1.062. Conservative: noise zone or slightly worse than 042A if instability bites.
