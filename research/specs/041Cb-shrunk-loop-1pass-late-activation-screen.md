# Spec 041Cb — shrunk loop (layers 4-5), NUM_LOOPS=1, frac=0.46

**Slug:** `041Cb-shrunk-loop-1pass-late-activation-screen`
**Created:** 2026-04-25
**Status:** READY

## Correction note

Original 041C used frac=0.43, landing at ~5665 steps (not 6000). Actual loop
throughput measured at 3277K tok/s across all 041 configs. Corrected frac=0.65:
780s no-loop × 5.457 + 420s loop × 4.165 ≈ 6000 steps.
**Branch:** `exp/039b-loop-band-activation-screen`
**Commit:** `5bbf12f`
**Links to:** `research/evaluations/040-no-loop-ablation.md`, `research/specs/041A-shrunk-loop-late-activation-screen.md`

## Hypothesis

Sibling of 041A. Same basin-crossing motivation: activate late so the model
reaches the lower-loss basin before recurrence kicks in. This arm tests
**shrunk layer range (4-5) AND shallow depth (1 extra pass)** — 13 effective
passes, nearly no-loop throughput (~4185K tok/s).

Two changes from baseline:
1. **LOOP_START=4, NUM_LOOPS=1**: 1 extra pass on layers 4-5 (13 total passes).
   Loop throughput ~3646K tok/s (~4.635 steps/s), only ~7% slower than no-loop.
2. **ENABLE_LOOPING_AT=0.43**: activate at 516s so total steps land at ~6000.

**Expected trajectory:**
- Phase 1 (0–516s): ~5.477 steps/s → ~2826 steps
- Phase 2 (516–1200s): ~4.635 steps/s → ~3170 recurrent steps
- Total: ~5996 steps

This is the lightest recurrence variant in the 041 family — minimal compute
overhead, maximum runway. If this wins, recurrence barely costs anything.

## Config diff

Four changes from the 039b baseline (commit `5bbf12f`):

```bash
NUM_LOOPS=1
LOOP_START=4
LOOP_END=5
ENABLE_LOOPING_AT=0.43
RECUR_ALPHA_ENABLED=0
```

`RECUR_ALPHA_ENABLED=0` required: recur_alpha/recur_beta calibrated for 3 loop
layers; with LOOP_START=4 those values map to wrong layers. Carry is neutral
(established by 039bK/bL).

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
export NUM_LOOPS=1 LOOP_START=4 LOOP_END=5 ENABLE_LOOPING_AT=0.46
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
export RUN_ID="041Cb-shrunk-loop-1pass-late-activation"

mkdir -p /workspace/runs/041Cb-shrunk-loop-1pass-late-activation-screen

torchrun --standalone --nproc_per_node=4 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py \
  >> /workspace/runs/041Cb-shrunk-loop-1pass-late-activation-screen/train.log 2>&1
```

## What to watch

- **Throughput Phase 1 (0–516s):** ~4308K tok/s
- **Loop activation:** around step ~2826, log shows `layer_loop:enabled`
- **Throughput Phase 2:** ~3646K tok/s (nearly no-loop speed)
- **Step at wallclock cap:** ~5996 steps
- **Activation spike:** smallest in the 041 family (1 pass, 2 layers)

## Acceptance

Baseline (039b): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641**
- **Noise zone**: 1.0641–1.0670 — run a second seed
- **Kill**: pre-quant EMA val_bpb ≥ **1.0670**
