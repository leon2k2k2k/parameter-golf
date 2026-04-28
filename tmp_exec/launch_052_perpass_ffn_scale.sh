#!/bin/bash
# Spec 052 — per-pass FFN scale on loop layers
# Learned [n_passes, n_loop_layers, model_dim] scale on MLP output, ones-init.
# 4608 params total. Identity at init → can only help if gradient finds signal.
# Based on LoopFormer finding: pass-phase signals give +1.93pp zero-shot at near-zero cost.
# Compare to 050A baseline pre-quant EMA bpb 1.06484.
# Accept: bpb <= 1.064. Kill: > 1.068.
set -euo pipefail

SHA="cc64dd4"
ARM="052-perpass-ffn-scale"
RUNDIR="/workspace/runs/052-perpass-ffn-scale-screen"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py"

# ── 1. Git setup ──────────────────────────────────────────────────────────────
WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# ── 2. Deps ───────────────────────────────────────────────────────────────────
pip install brotli python-minifier sentencepiece --break-system-packages -q

# ── 3. Output dir + inductor cache ────────────────────────────────────────────
mkdir -p "$RUNDIR"
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache

# ── 4. Config ─────────────────────────────────────────────────────────────────
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5
export LOOP_SCALE_INIT=recip LOOP_ITER_EMBEDS=1
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.99 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export SMEAR_GATE_ENABLED=1 SKIP_GATES_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export SEED=42 PHASED_TTT_NUM_PHASES=3 TTT_ENABLED=0

# ── Phase 0: inline prewarm ───────────────────────────────────────────────────
# New code → new compiled graph. ENABLE_LOOPING_AT=0.05 forces loop activation
# at step ~25, compiling both pre-loop and post-loop graph variants before Phase 1.
echo "[launch] Phase 0: inline prewarm (compile both graph variants)"
export ENABLE_LOOPING_AT=0.05
export MAX_WALLCLOCK_SECONDS=900
export RUN_ID="${ARM}-prewarm"
torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/prewarm.log" 2>&1
echo "[launch] Phase 0 done. Cache: $(du -sh /tmp/inductor_cache | cut -f1)"

# ── Phase 1: real 20-min screen ───────────────────────────────────────────────
echo "[launch] Phase 1: real screen (20 min, kernels warm)"
echo "[launch] Accept: pre-quant EMA bpb <= 1.064 | Kill: > 1.068"
export ENABLE_LOOPING_AT=0.35
export MAX_WALLCLOCK_SECONDS=1200
export RUN_ID="${ARM}"
torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/train.log" 2>&1

echo "[launch] 052 per-pass FFN scale screen done."
