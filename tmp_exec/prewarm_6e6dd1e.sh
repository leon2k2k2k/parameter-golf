#!/bin/bash
# Prewarm for spec 051 PPM-D — commit 6e6dd1e
# New kernel shapes: [1024,512] matmuls (half-width MLP) in loop layers.
# MUST run before any 051 arm. Stash cache after, restore on each arm pod.
#
# After this script exits, from LOCAL MACHINE:
#   bash tmp_exec/cache_stash.sh <host> <port> 6e6dd1e
#
# On each arm pod before training:
#   bash /workspace/parameter-golf/tmp_exec/restore_cache_local.sh 6e6dd1e
set -euo pipefail

SHA="6e6dd1e"
LOGDIR="/workspace/runs/_prewarm/${SHA}"
mkdir -p "$LOGDIR"

pip install brotli python-minifier sentencepiece --break-system-packages -q

WORKTREE="/workspace/pg-prewarm-${SHA}"
if [ ! -d "$WORKTREE" ]; then
  git -C /workspace/parameter-golf worktree add --detach "$WORKTREE" "$SHA"
fi
cd "$WORKTREE"
git checkout "$SHA"

TRAIN="$WORKTREE/records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py"

export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache

# --- Config: exact 050A settings ---
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
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=0
export SEED=42 PHASED_TTT_NUM_PHASES=3

# --- Prewarm-only overrides ---
# ENABLE_LOOPING_AT=0.05 forces loop activation at step ~25 (5% of 20min run)
# so the half-width [1024,512] kernels get autotuned within MAX_WALLCLOCK_SECONDS=900
export ENABLE_LOOPING_AT=0.05
export MAX_WALLCLOCK_SECONDS=900  # 15 min: enough for cold compile + loop activation + verify
export TTT_ENABLED=0
export RUN_ID="prewarm-6e6dd1e"
LOGFILE="$LOGDIR/stage1.log"

echo "[prewarm] 051 PPM-D — commit ${SHA}"
echo "[prewarm] ENABLE_LOOPING_AT=0.05 → loop activates ~step 25, forces [1024,512] kernel compile"
echo "[prewarm] MAX_WALLCLOCK_SECONDS=900 — expect ~15-20 min cold (new kernel shapes)"
echo "[prewarm] Log: ${LOGFILE}"

torchrun --standalone --nproc_per_node=4 "$TRAIN" >> "$LOGFILE" 2>&1

TOK=$(grep "^100/20000 train_loss" "$LOGFILE" | tail -1 | grep -o 'tok/s: [0-9]*' | grep -o '[0-9]*' || true)
echo "[prewarm] Done. tok/s at step 100: ${TOK:-UNKNOWN}"
echo "[prewarm] Cache size: $(du -sh /tmp/inductor_cache | cut -f1)"

if [ -n "$TOK" ] && [ "$TOK" -lt 4300000 ]; then
  echo "[prewarm] WARNING: tok/s ${TOK} < 4,300,000 — pod may be slow, check before committing"
else
  echo "[prewarm] OK: throughput looks good"
fi

echo ""
echo "[prewarm] → From LOCAL MACHINE, stash the cache:"
echo "    bash tmp_exec/cache_stash.sh <host> <port> 6e6dd1e"
echo ""
echo "[prewarm] → On each arm pod, restore before training:"
echo "    bash /workspace/parameter-golf/tmp_exec/restore_cache_local.sh 6e6dd1e"
