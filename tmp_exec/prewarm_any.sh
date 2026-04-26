#!/bin/bash
# Generic full-autotune pre-warm for any code commit.
# Run ONCE on a fresh pod before any arm. Then stash with cache_stash.sh.
#
# Usage:
#   bash prewarm_any.sh <commit-sha> <stage1-env-overrides> [stage2-env-overrides ...]
#
#   Each "env-overrides" arg is a space-separated list of KEY=VALUE pairs applied on top
#   of the base config for that stage. Stages run sequentially — later stages hit mostly
#   cache (only the newly branched torch.compile path needs fresh compilation).
#
# Example (fc54262, two graph variants):
#   bash tmp_exec/prewarm_any.sh fc54262 \
#     "LOOP_ITER_EMBEDS=1 LOOP_PER_PASS_RESID_MIX=0" \
#     "LOOP_ITER_EMBEDS=1 LOOP_PER_PASS_RESID_MIX=1"
#
# After this script exits, run from LOCAL MACHINE:
#   bash tmp_exec/cache_stash.sh <host> <port> <commit-sha>
#
# WHY NOT TRITON_AUTOTUNE_NUM_RUNS=1:
#   That flag picks first-candidate kernels. At 4xH100 full model scale it costs ~6%
#   throughput (~250 steps in a 20-min run), biasing LR warmdown timing and EMA quality.
#   Pre-warming bakes optimal kernels into the stashed cache — subsequent arms restore
#   the cache and get full throughput with zero recompile risk.
#
# Base config update policy:
#   This base config covers specs 045+. If the model architecture changes (new record
#   directory, different NUM_LAYERS/MODEL_DIM, new feature flags), update the base
#   config block below to match. The commit-sha arg just pins the code version; the
#   config must match what the actual arms will run.

set -euo pipefail

SHA="${1:?Usage: prewarm_any.sh <commit-sha> <stage1-env> [stage2-env ...]}"
shift  # remaining args are per-stage env override strings

if [ $# -eq 0 ]; then
  echo "[prewarm] ERROR: at least one stage env-override arg required."
  echo "[prewarm] Example: bash prewarm_any.sh fc54262 \"LOOP_ITER_EMBEDS=1 LOOP_PER_PASS_RESID_MIX=0\""
  exit 1
fi

LOGDIR="/workspace/runs/_prewarm/${SHA}"
mkdir -p "$LOGDIR"

# --- Deps ---
pip install brotli python-minifier sentencepiece --break-system-packages -q

# --- Worktree: commit-specific path so concurrent pods don't clobber each other ---
# /workspace/pg-prewarm is NOT used — shared path would let a second pod's
# `git checkout <other-sha>` silently switch the directory mid-run.
WORKTREE="/workspace/pg-prewarm-${SHA}"
if [ ! -d "$WORKTREE" ]; then
  git -C /workspace/parameter-golf worktree add --detach "$WORKTREE" "$SHA"
fi
cd "$WORKTREE"
git checkout "$SHA"

# Path to train_gpt.py — update this if the baseline record directory changes.
TRAIN="$WORKTREE/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py"

# --- Inductor cache ---
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache

# --- Base config (specs 045+) ---
# Architecture and feature-flag fields affect compiled graphs.
# Hyperparameters (LR, momentum, etc.) do not — included for a valid run.
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
# Prewarm-only: skip GPTQ eval, cap at 5 min of training (enough to verify tok/s + flush cache)
export TRAINING_ONLY_SCREEN=1
export MAX_WALLCLOCK_SECONDS=300

# --- Run each stage ---
STAGE=0
ALL_TOK=""
for EXTRA_VARS in "$@"; do
  STAGE=$((STAGE + 1))
  LOGFILE="$LOGDIR/stage${STAGE}.log"
  export RUN_ID="prewarm-${SHA:0:7}-stage${STAGE}"

  echo ""
  echo "[prewarm] ====== STAGE ${STAGE} / $# : ${EXTRA_VARS} ======"
  if [ "$STAGE" -eq 1 ]; then
    echo "[prewarm] Full autotune — expect ~15 min cold compile, then 5 min training."
  else
    echo "[prewarm] Mostly cache hits — only new graph variant needs compile (~2 min)."
  fi

  # Apply stage-specific env overrides on top of base config
  eval "export ${EXTRA_VARS}"

  torchrun --standalone --nproc_per_node=4 "$TRAIN" >> "$LOGFILE" 2>&1

  TOK=$(grep "^100/20000 train_loss" "$LOGFILE" | tail -1 | \
        grep -o 'tok/s: [0-9]*' | grep -o '[0-9]*' || true)
  echo "[prewarm] Stage ${STAGE} done. tok/s at step 100: ${TOK:-UNKNOWN}"
  echo "[prewarm] Cache: $(du -sh /tmp/inductor_cache | cut -f1)"

  if [ -n "$TOK" ]; then
    if [ "$TOK" -lt 4300000 ]; then
      echo "[prewarm] WARNING: tok/s ${TOK} < 4,300,000 — pod may be slow or cache not optimal."
    else
      echo "[prewarm] OK: tok/s ${TOK} >= 4,300,000"
    fi
    ALL_TOK="${ALL_TOK} Stage${STAGE}=${TOK}"
  fi
done

echo ""
echo "[prewarm] ===== PRE-WARM COMPLETE (${STAGE} stage(s)) ====="
echo "[prewarm] Throughput summary:${ALL_TOK}"
echo "[prewarm] Final cache: $(du -sh /tmp/inductor_cache | cut -f1)"
echo ""
echo "[prewarm] → From LOCAL MACHINE, stash the cache:"
echo "    bash tmp_exec/cache_stash.sh <host> <port> ${SHA}"
echo ""
echo "[prewarm] → On each subsequent arm pod:"
echo "    bash tmp_exec/restore_cache_local.sh ${SHA}"
echo "    export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache"
echo "    # Do NOT set TRITON_AUTOTUNE_NUM_RUNS=1"
echo "    # Verify tok/s >= 4,300,000 at step 100 before committing to a full 20-min run"
