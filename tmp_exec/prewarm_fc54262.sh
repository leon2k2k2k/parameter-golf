#!/bin/bash
# Full-autotune pre-warm for fc54262 — run ONCE on a fresh pod before any arm.
#
# Root cause of 6% throughput loss: TRITON_AUTOTUNE_NUM_RUNS=1 picks first-candidate
# kernels. At 4×H100 full scale this consistently costs ~250 training steps vs baseline.
# Proof: AC-fix log shows Run 1 (no flag) = 4,338,577 tok/s vs Run 2 (flag) = 4,085,843.
#
# This script runs two sequential pre-warm passes with full autotune:
#   Stage 1: LOOP_ITER_EMBEDS=1          → compiles graphs for AC-fix / A2 / G
#   Stage 2: LOOP_ITER_EMBEDS=1          → compiles additional graphs for H / GH
#            LOOP_PER_PASS_RESID_MIX=1     (loop_resid_mixes not None → different path)
#
# Both stages write to /tmp/inductor_cache (additive). Stage 2 is mostly cache hits.
# After this script exits, run from local machine:
#   bash tmp_exec/cache_stash.sh <host> <port> fc54262
#
# Expected: Stage 1 ~15 min compile + 5 min training. Stage 2 ~2 min compile + 5 min.
# Total: ~25 min on pod. Verify: tok/s ≥ 4.30M at step 100 on first arm after restore.
#
# DO NOT set TRITON_AUTOTUNE_NUM_RUNS=1 here or the point is defeated.

set -euo pipefail

export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache
pip install brotli python-minifier sentencepiece --break-system-packages -q

if [ ! -d /workspace/pg-prewarm-fc54262 ]; then
  git -C /workspace/parameter-golf worktree add --detach /workspace/pg-prewarm-fc54262 fc54262
fi
cd /workspace/pg-prewarm-fc54262
git checkout fc54262

TRAIN=/workspace/pg-prewarm-fc54262/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py
LOGDIR=/workspace/runs/045-loop-layer-improvements/prewarm-fc54262
mkdir -p "$LOGDIR"

# --- Shared config (identical to AC-fix/A2/G/H arms) ---
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
# Skip GPTQ eval — we only want the kernel cache, not val_bpb.
export TRAINING_ONLY_SCREEN=1
# 5 min of actual training after compile completes — enough to verify tok/s and flush cache.
export MAX_WALLCLOCK_SECONDS=300

# --- Stage 1: graphs for AC-fix / A2 / G (loop_iter_embeds not None) ---
echo "[prewarm] ====== STAGE 1: AC-fix/A2/G graphs (LOOP_ITER_EMBEDS=1) ======"
echo "[prewarm] Full autotune — expect ~15 min cold compile, then 5 min training."
export LOOP_ITER_EMBEDS=1
export LOOP_SCALE_INIT=ones   # init doesn't affect graph hash
export LOOP_LR_SCALE=off      # hooks don't affect compiled forward
export LOOP_PER_PASS_RESID_MIX=0
export RUN_ID="045-prewarm-stage1"

torchrun --standalone --nproc_per_node=4 "$TRAIN" \
  >> "$LOGDIR/stage1.log" 2>&1

S1_TOK=$(grep "^100/20000 train_loss" "$LOGDIR/stage1.log" | tail -1 | grep -o 'tok/s: [0-9]*' | grep -o '[0-9]*')
echo "[prewarm] Stage 1 done. tok/s at step 100: ${S1_TOK:-UNKNOWN}"
echo "[prewarm] Cache size: $(du -sh /tmp/inductor_cache | cut -f1)"

# --- Stage 2: additional graphs for H / GH (loop_resid_mixes not None) ---
echo "[prewarm] ====== STAGE 2: H/GH graphs (LOOP_PER_PASS_RESID_MIX=1) ======"
echo "[prewarm] Mostly cache hits — only new Block.forward path needs compile (~2 min)."
export LOOP_PER_PASS_RESID_MIX=1
export RUN_ID="045-prewarm-stage2"

torchrun --standalone --nproc_per_node=4 "$TRAIN" \
  >> "$LOGDIR/stage2.log" 2>&1

S2_TOK=$(grep "^100/20000 train_loss" "$LOGDIR/stage2.log" | tail -1 | grep -o 'tok/s: [0-9]*' | grep -o '[0-9]*')
echo "[prewarm] Stage 2 done. tok/s at step 100: ${S2_TOK:-UNKNOWN}"
echo "[prewarm] Cache size: $(du -sh /tmp/inductor_cache | cut -f1)"

echo ""
echo "[prewarm] ===== PRE-WARM COMPLETE ====="
echo "[prewarm] Both graph variants compiled with full autotune."
echo "[prewarm] Expected tok/s: ≥ 4,300,000. Got: Stage1=${S1_TOK:-?} Stage2=${S2_TOK:-?}"
echo ""
echo "[prewarm] Now run from LOCAL MACHINE:"
echo "  bash tmp_exec/cache_stash.sh <host> <port> fc54262"
echo ""
echo "[prewarm] On all subsequent arm runs:"
echo "  1. cache_restore.sh <host> <port> fc54262"
echo "  2. Set TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache"
echo "  3. Do NOT set TRITON_AUTOTUNE_NUM_RUNS=1"
echo "  4. Verify tok/s >= 4.30M at step 100 before committing full 20-min run"
