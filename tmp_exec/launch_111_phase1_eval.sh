#!/bin/bash
# Eval-only launcher for spec 060B/D/E/F (RESUME_FROM_CKPT mode).
#
# Skips train_model(), loads final_model.pt from a 060A-or-later run,
# runs pre-quant eval → GPTQ + serialize → post-quant eval → TTT eval.
# Cost: ~$1-3, ~5-15 min.
#
# Usage:
#   SEED=42 RUN_LABEL=seed_42_X \
#     RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
#     [override env vars per-spec, e.g. ATTN_CLIP_SIGMAS=12.5] \
#     bash launch_060_eval.sh
#
# Required env vars: SEED, RUN_LABEL, RESUME_FROM_CKPT.

set -euo pipefail

SEED="${SEED:?Set SEED}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:?Set RESUME_FROM_CKPT (path to final_model.pt)}"

[ -f "$RESUME_FROM_CKPT" ] || { echo "ERR: ckpt not found: $RESUME_FROM_CKPT"; exit 1; }

# ── Identity ──────────────────────────────────────────────────────────────
SHA="${SHA:-e9da01a}"   # default: spec 060A's commit
ARM="${ARM:-111-anderson-phase1-eval}"
RUNDIR="/workspace/runs/${ARM}/${RUN_LABEL}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py}"

# ── Setup ────────────────────────────────────────────────────────────────
WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")
which lrzip > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y lrzip) 2>&1 | tail -5
pip install brotli sentencepiece --break-system-packages -q

mkdir -p "$RUNDIR"
cd "$RUNDIR"

# Copy the .pt into RUNDIR so deserialize sees it
cp -n "$RESUME_FROM_CKPT" "$RUNDIR/final_model.pt" 2>/dev/null || true

export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache

# ── Default config (matches 060A; override anything via env before calling) ──
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
export BETA1=0.9 BETA2=0.99 EMA_DECAY=0.9965
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
export ITERATIONS=20000 WARMDOWN_FRAC=0.85 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
export CASEOPS_ENABLED=1 COMPRESSOR=pergroup SMEAR_GATE_ENABLED=1 SKIP_GATES_ENABLED=1   # pergroup = per-group lrzip; saves ~280 KB vs brotli (#1855's compressor)
export SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_ENABLED=0 ATTN_OUT_GATE_ENABLED=0
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=14.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export TTT_ENABLED=1 PHASED_TTT_ENABLED=3 PHASED_TTT_NUM_PHASES=3
export PHASED_TTT_PREFIX_DOCS=2500 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80

# RESUME_FROM_CKPT magic — train_gpt.py should detect this and skip training.
export RESUME_FROM_CKPT="$RESUME_FROM_CKPT"

# ── Spec 111 phase 1 — Anderson at eval on 060A trained checkpoint ───────
# Critical: these must be exported BEFORE torchrun, not after.
export ANDERSON_ENABLED=1
export ANDERSON_HISTORY=2
export ANDERSON_BETA=1.0
export ANDERSON_REGULARIZATION=1e-6
echo "[launch_111_phase1_eval] ANDERSON_ENABLED=${ANDERSON_ENABLED} HISTORY=${ANDERSON_HISTORY} BETA=${ANDERSON_BETA}"

# Per-run identity
export SEED="$SEED"
export MAX_WALLCLOCK_SECONDS=900   # eval-only; gives slack
export RUN_ID="${ARM}-${RUN_LABEL}"
export ARTIFACT_DIR="$RUNDIR"

# ── Echo non-default overrides ────────────────────────────────────────────
echo "=========================================================================="
echo "[eval] spec ${ARM} / SEED=${SEED} / RUN_LABEL=${RUN_LABEL}"
echo "[eval] RESUME from: ${RESUME_FROM_CKPT}"
echo "[eval] Overrides:"
env | grep -E "^(MLP_CLIP_SIGMAS|ATTN_CLIP_SIGMAS|EMBED_CLIP_SIGMAS|LQER_|EMBED_BITS|DEPLOY_TIME_REPAIR_|SPINQUANT_)=" | sort
echo "=========================================================================="

# ── Run (4×H100 eval-only) ────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/eval.log" 2>&1

echo "[eval] done."

# ── Verify checkpoint ─────────────────────────────────────────────────────
PTZ_PATH="${RUNDIR}/final_model.int6.ptz"
if [ -f "$PTZ_PATH" ]; then
    SIZE=$(stat -c%s "$PTZ_PATH")
    # CAP IS ON TOTAL SUBMISSION (.int6.ptz + ~32 KB compressed code), NOT on .int6.ptz alone.
    TOTAL_SUB=$(grep -oE "Total submission size quantized[+:][^ ]+ [0-9]+ bytes" "${RUNDIR}/eval.log" | tail -1 | grep -oE '[0-9]+' | tail -1)
    echo "[eval] OK: int6.ptz = ${SIZE} bytes alone"
    if [ -n "$TOTAL_SUB" ]; then
        echo "[eval] TOTAL submission = ${TOTAL_SUB} bytes (cap 16,000,000; margin $((16000000 - TOTAL_SUB)))"
        if [ "$TOTAL_SUB" -gt 16000000 ]; then
            echo "[eval] FAIL: total submission OVER CAP by $((TOTAL_SUB - 16000000)) bytes"
            exit 3
        fi
    fi
else
    echo "[eval] WARN: missing int6.ptz"
fi

# ── Surface result ────────────────────────────────────────────────────────
echo "=========================================================================="
grep -E "diagnostic.*val_bpb|val_loss:|TTT|sliding|^bytes" "${RUNDIR}/eval.log" | tail -10 || true

