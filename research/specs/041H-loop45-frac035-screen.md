# Spec 041H — loop layers 4-5, NUM_LOOPS=2, frac=0.35 (baseline timing)

**Slug:** `041H-loop45-frac035-screen`
**Created:** 2026-04-26
**Status:** READY
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/ideas/041-late-loop-activation-basin-crossing.md`

## Hypothesis

We concluded "frac=0.46 is optimal" from the 041 arc — but that comparison always
changed frac AND N simultaneously. We never isolated the effect of N=15 at frac=0.35.

**Key insight:** shrinking loop width (3 layers → 2 layers) makes each loop-phase
second worth more loop steps. At frac=0.46 that gain was cancelled by later activation
(fewer loop-phase seconds). At frac=0.35 (same activation as baseline) it's pure upside:

| Config | N | Loop phase | Loop steps/s | Loop steps |
|--------|---|-----------|-------------|-----------|
| Baseline (3-5, NL=2) | 17 | 780s | 3.717 | 2899 |
| **041H (4-5, NL=2)** | **15** | **780s** | **4.167** | **3250** |

Same 420s no-loop phase, same loop activation timing. But 041H runs 12% faster in
the loop phase → **+351 loop steps vs baseline**.

Predicted val_bpb: 1.06514 - 351×2.2e-6 = **~1.06437** (beats baseline by 0.00077).

Risk: extrapolating past 2867 loop steps (the max observed). Diminishing returns
possible. Also: 2-layer loop may provide less quality per step than 3-layer at this
activation timing — but 041A showed 2 vs 3 loop layers barely changed val_bpb at
matched loop steps.

## Config diff

Two changes from the 039b baseline (commit `5bbf12f`):

```bash
LOOP_START=4
LOOP_END=4
ENABLE_LOOPING_AT=0.35
RECUR_ALPHA_ENABLED=0
```

Same NUM_LOOPS=2 as baseline. RECUR_ALPHA_ENABLED=0 required: recur_alpha/recur_beta
calibrated for 3 loop layers (3-5); shape mismatch with LOOP_START=4.

## Regime

- `4×H100`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=1`
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
export NUM_LOOPS=2 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.35
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
export SEED=42 MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=1
export RUN_ID="041H-loop45-frac035"

mkdir -p /workspace/runs/041H-loop45-frac035-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/041H-loop45-frac035-screen/train.log 2>&1
```

## What to watch

- **Loop activation:** step ~2308 (same as baseline), log shows `layer_loop:enabled`
- **Throughput Phase 2:** ~4167 steps/s (vs baseline's ~3717 — visibly faster loop phase)
- **Step at wallclock cap:** ~5558 steps
- **Loop steps:** ~3250 (vs baseline's ~2899)

## Acceptance

Baseline (039b): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641**
- **Noise zone**: 1.0641–1.0670 — run second seed
- **Kill**: pre-quant EMA val_bpb ≥ **1.0670**

## Prediction

~1.06437 based on +351 loop steps at 2.2e-6 rate. First extrapolation past 2867 loop
steps — diminishing returns possible. Key signal: does loop-phase tok/s land at
~3277K (confirming N=15 throughput) and does loop activate at step ~2308?
