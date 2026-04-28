#!/bin/bash
# Spec 060A — port PR #1855 as new research baseline (1-seed validation).
#
# NO PREWARM (per user instruction). Cold compile inside the 600s wallclock.
#
# Usage: SEED=42 RUN_LABEL=seed_42 bash launch_060A_run.sh
#
# Required env vars: SEED, RUN_LABEL.
# All other env vars are set below.

set -euo pipefail

# ── Parameters from caller ────────────────────────────────────────────────
SEED="${SEED:?Set SEED env var}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL (e.g. seed_42)}"

# ── Pinned commit + identity ──────────────────────────────────────────────
SHA="da50cd6"
ARM="060A-1855-port"
RUNDIR="/workspace/runs/${ARM}/${RUN_LABEL}"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py"

# ── Git setup ─────────────────────────────────────────────────────────────
WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# ── System deps ──────────────────────────────────────────────────────────
# lrzip is REQUIRED for #1855's per-group compressor (_lrzip_compress).
# brotli + sentencepiece are pip deps. Install all unconditionally; cheap if already present.
which lrzip > /dev/null 2>&1 || apt-get install -y lrzip 2>&1 | tail -3
pip install brotli python-minifier sentencepiece --break-system-packages -q

# ── Output dir ────────────────────────────────────────────────────────────
mkdir -p "$RUNDIR"
cd "$RUNDIR"

# ── Inductor cache (cold compile; tolerate if cache restore fails) ────────
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache
# Try to restore stashed cache for this SHA if one happens to exist.
if bash /workspace/parameter-golf/tmp_exec/restore_cache_local.sh "$SHA" 2>/dev/null; then
    echo "[launch] inductor cache restored for ${SHA}"
else
    echo "[launch] no stashed cache for ${SHA}; cold compile inside 600s budget"
fi

# ── Config (PR #1855's defaults made explicit) ────────────────────────────
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'

# Architecture (1797 base, unchanged)
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
# Loop
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
# Optimizer (BETA2=0.99 is 1855 default)
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.99 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
# WARMDOWN_FRAC=0.85 is 1855 default
export ITERATIONS=20000 WARMDOWN_FRAC=0.85 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
# Features (1855 stack)
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export SMEAR_GATE_ENABLED=1
export SKIP_GATES_ENABLED=1
# SPARSE_ATTN_GATE_SCALE=0.5 is 1855 default
export SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
# Quantization clip values (MLP_CLIP=11.5 + EMBED_CLIP=14.0 are 1855 defaults)
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=14.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
# TTT (1855 defaults: PHASED_TTT_PREFIX_DOCS=2500, TTT_BETA2=0.99, TTT_WD=0.5, TTT_LORA_RANK=80)
export TTT_ENABLED=1
export PHASED_TTT_ENABLED=3
export PHASED_TTT_NUM_PHASES=3
export PHASED_TTT_PREFIX_DOCS=2500
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80

# ── Per-run identity ──────────────────────────────────────────────────────
export SEED="$SEED"
export MAX_WALLCLOCK_SECONDS=600
export RUN_ID="${ARM}-${RUN_LABEL}"
export ARTIFACT_DIR="$RUNDIR"   # final_model.{pt,int6.ptz} land here

# ── Echo final config ─────────────────────────────────────────────────────
echo "=========================================================================="
echo "[launch] spec 060A / arm=${ARM} / SHA=${SHA} / SEED=${SEED} / RUN_LABEL=${RUN_LABEL}"
echo "[launch] MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}  ITERATIONS=${ITERATIONS}"
echo "[launch] WARMDOWN_FRAC=${WARMDOWN_FRAC}  MLP_CLIP_SIGMAS=${MLP_CLIP_SIGMAS}  EMBED_CLIP_SIGMAS=${EMBED_CLIP_SIGMAS}"
echo "[launch] TTT: PREFIX_DOCS=${PHASED_TTT_PREFIX_DOCS}  BETA2=${TTT_BETA2}  WD=${TTT_WEIGHT_DECAY}  LORA_RANK=${TTT_LORA_RANK}"
echo "[launch] ARTIFACT_DIR=${ARTIFACT_DIR}  (final_model.{pt,int6.ptz} land here)"
echo "[launch] WORKTREE=${WORKTREE}"
echo "=========================================================================="

# ── Run ───────────────────────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=8 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/train.log" 2>&1

echo "[launch] spec 060A ${RUN_LABEL} done."

# ── Verify + protect checkpoints ──────────────────────────────────────────
# Both artifacts MUST exist; spec 060B+ depends on these for downstream stacking.
PT_PATH="${RUNDIR}/final_model.pt"
PTZ_PATH="${RUNDIR}/final_model.int6.ptz"

if [ -f "$PT_PATH" ] && [ -f "$PTZ_PATH" ]; then
    PT_SIZE=$(stat -c%s "$PT_PATH")
    PTZ_SIZE=$(stat -c%s "$PTZ_PATH")
    echo "[launch] OK: BOTH checkpoints saved in ${RUNDIR}/"
    echo "[launch]   final_model.pt        = ${PT_SIZE} bytes  (~$((PT_SIZE / 1024 / 1024)) MB pre-quant)"
    echo "[launch]   final_model.int6.ptz  = ${PTZ_SIZE} bytes (cap 16,000,000; margin $((16000000 - PTZ_SIZE)))"
    ls -la "${RUNDIR}/final_model"*
    # Make read-only to prevent accidental overwrite by future runs in same dir.
    chmod a-w "$PT_PATH" "$PTZ_PATH" 2>/dev/null || true
    echo "[launch] Checkpoints chmod'd a-w to protect from overwrite."
else
    echo "[launch] FAIL: missing final_model artifacts in ${RUNDIR}"
    echo "[launch]   exists: $([ -f "$PT_PATH" ] && echo yes || echo NO) final_model.pt"
    echo "[launch]   exists: $([ -f "$PTZ_PATH" ] && echo yes || echo NO) final_model.int6.ptz"
    ls -la "${RUNDIR}/" || true
    exit 2
fi

# ── Surface key result lines ──────────────────────────────────────────────
echo "=========================================================================="
echo "[launch] KEY RESULT LINES:"
grep -E "stopping_early|diagnostic.*val_bpb|val_loss:|Total submission size|TTT|sliding|^bytes" "${RUNDIR}/train.log" | tail -20 || true
