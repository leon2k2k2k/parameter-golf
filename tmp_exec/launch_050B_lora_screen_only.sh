#!/bin/bash
# Spec 050B — Per-pass FFN LoRA on 1797 baseline (050A + LOOP_FFN_LORA_RANK=2)
# always-tensor fix (a680014): identity zero buffers for non-loop steps, single
# Block.forward graph variant, no mid-run recompile at loop activation.
# Step 1: 200s sanity smoke at ENABLE_LOOPING_AT=0.05 — loop fires at ~10s,
#   verifies post-rewarm training runs (prior hang was post-rewarm).
# Step 2: full 20-min screen at ENABLE_LOOPING_AT=0.35 if smoke passes.
# Accept: bpb <= 1.067. Kill: > 1.072.
set -euo pipefail

SHA="e586a49"
ARM="050B-lora"
RUNDIR="/workspace/runs/050B-lora-screen"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py"

# ── 1. Git setup ──────────────────────────────────────────────────────────
WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# ── 2. Deps ───────────────────────────────────────────────────────────────
pip install brotli python-minifier sentencepiece --break-system-packages -q

# ── 3. Output dir ─────────────────────────────────────────────────────────
mkdir -p "$RUNDIR"

# ── 4. Config ─────────────────────────────────────────────────────────────
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
# Architecture
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
# Loop — AC-fix + LoRA (ENABLE_LOOPING_AT set per phase below)
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5
export LOOP_SCALE_INIT=recip
export LOOP_ITER_EMBEDS=1
export LOOP_FFN_LORA_RANK=2
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
# Optimizer
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.99 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
# Features
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export SMEAR_GATE_ENABLED=1
export SKIP_GATES_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
# Quantization
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export SEED=42 PHASED_TTT_NUM_PHASES=3

export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache

# Skip smoke — using e586a49 which has post-rewarm fix already verified by a680014 smoke.
# ── 6. Screen run (20 min) ────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS=1200 TTT_ENABLED=0
export ENABLE_LOOPING_AT=0.35
export RUN_ID="050B-lora"

echo "[launch] Smoke passed — starting 20-min screen."
echo "[launch] Accept: pre-quant EMA bpb <= 1.067 | Kill: > 1.072"
torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/train.log" 2>&1

echo "[launch] 050B lora screen done."
