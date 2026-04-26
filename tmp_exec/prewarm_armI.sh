#!/bin/bash
# Pre-warm for SHA 7926027 (Lever I: LOOP_REVERSE_LAST_PASS).
# New commit — cache is cold. Must run before launch_045_armI.sh.
#
# Only needs stage 1: LOOP_REVERSE_LAST_PASS=1 produces one new graph variant
# (the reversed decoder index sequence). Stage 2 (LOOP_PER_PASS_RESID_MIX) not needed.
#
# Expected: ~15 min cold compile + 5 min training. Total ~20 min.
# After this script exits, run from LOCAL MACHINE:
#   bash tmp_exec/cache_stash.sh <host> <port> 7926027
set -euo pipefail

SHA="7926027"
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache

# Seed cache from 7c6ac51 — pre-loop graphs are identical, only reversed-pass
# decoder graph is new. This saves ~10 min vs cold compile.
if [ -d /workspace/.inductor_cache_7c6ac51 ]; then
  echo "[prewarm] seeding from 7c6ac51 cache (pre-loop graphs reusable)..."
  rsync -a /workspace/.inductor_cache_7c6ac51/ /tmp/inductor_cache/
  echo "[prewarm] seeded: $(du -sh /tmp/inductor_cache | cut -f1)"
else
  echo "[prewarm] WARNING: 7c6ac51 cache not found — starting cold (will take ~15 min longer)"
fi

pip install brotli python-minifier sentencepiece --break-system-packages -q

if [ ! -d /workspace/pg-prewarm-${SHA} ]; then
  git -C /workspace/parameter-golf fetch fork exp/045-loop-layer-improvements
  git -C /workspace/parameter-golf worktree add --detach /workspace/pg-prewarm-${SHA} ${SHA}
fi
cd /workspace/pg-prewarm-${SHA}

TRAIN=/workspace/pg-prewarm-${SHA}/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py
LOGDIR=/workspace/runs/045-loop-layer-improvements/prewarm-${SHA}
mkdir -p "$LOGDIR"

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
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=0
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=1 RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5 SLOPE_WARMDOWN=-1.0
export SEED=42 PHASED_TTT_ENABLED=3 PHASED_TTT_NUM_PHASES=3
export TRAINING_ONLY_SCREEN=1
export MAX_WALLCLOCK_SECONDS=300
export LOOP_SCALE_INIT=recip
export LOOP_REVERSE_LAST_PASS=1
export RUN_ID="045-prewarm-armI"

echo "[prewarm] ====== ARM I PRE-WARM (SHA ${SHA}) ======"
echo "[prewarm] Full autotune for LOOP_REVERSE_LAST_PASS=1 graph. Expect ~15 min compile."

torchrun --standalone --nproc_per_node=4 "$TRAIN" \
  >> "$LOGDIR/stage1.log" 2>&1

TOK=$(grep "^100/20000 train_loss" "$LOGDIR/stage1.log" | tail -1 | grep -o 'tok/s: [0-9]*' | grep -o '[0-9]*')
echo "[prewarm] Done. tok/s at step 100: ${TOK:-UNKNOWN}"
echo "[prewarm] Cache size: $(du -sh /tmp/inductor_cache | cut -f1)"
echo ""
echo "[prewarm] Now run from LOCAL MACHINE:"
echo "  bash tmp_exec/cache_stash.sh <host> <port> ${SHA}"
