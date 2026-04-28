#!/bin/bash
# Spec 055 — 050 baseline NN + tuned PPM-D byte mixture (anti-hijack) at eval.
# Single train_gpt.py: training and post-training in same file.
#
# Usage: SEED=42 MAX_WALLCLOCK_SECONDS=600 RUN_LABEL=seed_42 bash launch_055_run.sh
#   - For smoke: SEED=42 MAX_WALLCLOCK_SECONDS=120 RUN_LABEL=smoke
#   - For real: SEED=42 MAX_WALLCLOCK_SECONDS=600 RUN_LABEL=seed_42  (and 7, 1337)
#
# Required env vars: SEED, MAX_WALLCLOCK_SECONDS, RUN_LABEL
# All other env vars are set below.

set -euo pipefail

# ── Parameters from caller ────────────────────────────────────────────────
SEED="${SEED:?Set SEED env var}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:?Set MAX_WALLCLOCK_SECONDS}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL (e.g. smoke or seed_42)}"

# ── Pinned commit + identity ──────────────────────────────────────────────
SHA="c27be23"
ARM="055-050-with-ppm-fullrun"
RUNDIR="/workspace/runs/${ARM}/${RUN_LABEL}"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py"

# ── Git setup ─────────────────────────────────────────────────────────────
WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# ── Deps (container disk → reinstall every pod start) ─────────────────────
pip install brotli python-minifier sentencepiece --break-system-packages -q

# ── Output dir ────────────────────────────────────────────────────────────
mkdir -p "$RUNDIR"
cd "$RUNDIR"

# ── Config — 050 BASELINE (verbatim from launch_050_baseline.sh) ──────────
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'

# Architecture (exactly 050 baseline)
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
# Loop
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
# Optimizer
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.99 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
# Features (050 stack: smear + lqer + sparse attn gate; no recur_alpha, no mlp_outer_activation)
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export SMEAR_GATE_ENABLED=1
export SKIP_GATES_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
# Quantization (used for full eval since PREQUANT_ONLY=0)
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
# TTT off (we use PPM not TTT)
export PHASED_TTT_NUM_PHASES=3
export TTT_ENABLED=0

# ── PPM post-training config (NEW for spec 055) ───────────────────────────
export PPM_NATIVE_ENABLED=1
export PPM_ORDER=4
export PPM_LAMBDA_HI=0.9
export PPM_LAMBDA_LO=0.05
export PPM_CONF_THRESHOLD=0.76          # tuned (vs 1850's 0.9)
export PPM_NN_SKIP_THR_NATS=0.277       # = 0.40 bits, anti-hijack
export PPM_LOG_CACHE_SIZE=1048576
export PPM_OMP_THREADS=8
export PPM_OMP_CHUNK_TOKENS=4194304     # OMP-chunked default

# ── Per-run identity ──────────────────────────────────────────────────────
export SEED="$SEED"
export MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS"
export RUN_ID="${ARM}-${RUN_LABEL}"
export ARTIFACT_DIR="$RUNDIR"   # so final_model.pt + final_model.int6.ptz land here

# ── Inductor cache (cold compile if no stash; warm after first run) ──────
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache
# Try to restore stashed cache; tolerate missing (cold compile is fine for spec 055)
if bash /workspace/parameter-golf/tmp_exec/restore_cache_local.sh "$SHA" 2>/dev/null; then
    echo "[launch] inductor cache restored from /workspace/.inductor_cache_${SHA}"
else
    echo "[launch] no stashed cache for ${SHA}; will compile cold (first run only)"
    # Try fallback: copy from 050 baseline cache (same training graph, our patch is eval-time only)
    if [ -d "/workspace/.inductor_cache_f5b8af8" ]; then
        echo "[launch] copying 050 baseline cache (f5b8af8) since training graph is identical"
        rsync -a /workspace/.inductor_cache_f5b8af8/ /tmp/inductor_cache/
    fi
fi

# ── Echo final config (sanity) ─────────────────────────────────────────────
echo "=========================================================================="
echo "[launch] spec 055 / arm=${ARM} / SHA=${SHA} / SEED=${SEED} / RUN_LABEL=${RUN_LABEL}"
echo "[launch] MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}  ITERATIONS=${ITERATIONS}"
echo "[launch] TTT_ENABLED=${TTT_ENABLED}  PPM_NATIVE_ENABLED=${PPM_NATIVE_ENABLED}"
echo "[launch] PPM_CONF_THRESHOLD=${PPM_CONF_THRESHOLD}  PPM_NN_SKIP_THR_NATS=${PPM_NN_SKIP_THR_NATS}"
echo "[launch] ARTIFACT_DIR=${ARTIFACT_DIR}  (final_model.{pt,int6.ptz} land here)"
echo "[launch] RUNDIR=${RUNDIR}"
echo "[launch] WORKTREE=${WORKTREE}"
echo "=========================================================================="

# ── Run ───────────────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=8 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/train.log" 2>&1

echo "[launch] spec 055 ${RUN_LABEL} done."

# ── Verify checkpoints exist (safety per user request) ────────────────────
if [ -f "${RUNDIR}/final_model.pt" ] && [ -f "${RUNDIR}/final_model.int6.ptz" ]; then
    echo "[launch] OK: both final_model.pt and final_model.int6.ptz saved in ${RUNDIR}"
    ls -la "${RUNDIR}/final_model"*
else
    echo "[launch] WARN: missing final_model artifacts in ${RUNDIR}"
    ls -la "${RUNDIR}/" || true
fi

# ── Surface PPM result (if PPM ran) ───────────────────────────────────────
grep -E '^ppm_native_submission_val_bpb:|^ppm_full_native ' "${RUNDIR}/train.log" | tail -2 || true
