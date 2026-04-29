#!/bin/bash
# Spec 060A on 4×H100 (when 8×H100 unavailable). Per execution memo:
# substitute --nproc=4, MAX_WALLCLOCK=1200 (matched FLOPs to 8H × 600s),
# GRAD_ACCUM_STEPS=2 (preserves global batch=786432, per-GPU mem unchanged).
#
# NOTE: result is NOT leaderboard-valid (training >600s). For research-side
# baseline validation only.
#
# Usage: SEED=42 RUN_LABEL=seed_42_4h bash launch_070_loop34.sh

set -euo pipefail

SEED="${SEED:?Set SEED}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"

SHA="e7ccda2"
ARM="071-loop-pattern-tent"
RUNDIR="/workspace/runs/${ARM}/${RUN_LABEL}"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py"

WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# System deps (lrzip apt cache may not be present on stock parameter-golf:latest)
which lrzip > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y lrzip) 2>&1 | tail -5
pip install brotli python-minifier sentencepiece --break-system-packages -q

mkdir -p "$RUNDIR"
cd "$RUNDIR"

export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache
bash /workspace/parameter-golf/tmp_exec/restore_cache_local.sh "$SHA" 2>/dev/null \
  && echo "[launch] cache restored" || echo "[launch] cold compile"

# ── Config (matches launch_060A_run.sh exactly except 4H adaptations) ────
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
export LOOP_PATTERN="2,3,4,5,4,5,6,5,6,7"
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.99 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS=20000 WARMDOWN_FRAC=0.85 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
export CASEOPS_ENABLED=1 COMPRESSOR=pergroup
export SMEAR_GATE_ENABLED=1 SKIP_GATES_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_ENABLED=0 ATTN_OUT_GATE_ENABLED=0
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=14.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export TTT_ENABLED=1 PHASED_TTT_ENABLED=3 PHASED_TTT_NUM_PHASES=3
export PHASED_TTT_PREFIX_DOCS=2500 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80

# ── 4H ADAPTATIONS ────────────────────────────────────────────────────────
export GRAD_ACCUM_STEPS=2          # global batch unchanged at 786432; halved GPU count → 2x accum
export MAX_WALLCLOCK_SECONDS=1200  # matched FLOPs to 8H × 600s

export SEED="$SEED"
export RUN_ID="${ARM}-${RUN_LABEL}"
export ARTIFACT_DIR="$RUNDIR"

echo "=========================================================================="
echo "[launch] 4H Spec 071 — Loop pattern tent (2,3,4,5,4,5,6,5,6,7) / SHA=${SHA} / SEED=${SEED} / RUN_LABEL=${RUN_LABEL}"
echo "[launch] WALLCLOCK=1200s / GRAD_ACCUM=2 (4H matched-FLOPs to 8H×600s)"
echo "[launch] NOT LEADERBOARD-VALID — research-side baseline only"
echo "=========================================================================="

torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/train.log" 2>&1

echo "[launch] 4H 060A done."

# ── Verify checkpoints + cap (TOTAL submission, not just .int6.ptz) ───────
PT_PATH="${RUNDIR}/final_model.pt"
PTZ_PATH="${RUNDIR}/final_model.int6.ptz"

if [ -f "$PT_PATH" ] && [ -f "$PTZ_PATH" ]; then
    PT_SIZE=$(stat -c%s "$PT_PATH")
    PTZ_SIZE=$(stat -c%s "$PTZ_PATH")
    TOTAL_SUB=$(grep -oE "Total submission size quantized[+:][^ ]+ [0-9]+ bytes" "${RUNDIR}/train.log" | tail -1 | grep -oE '[0-9]+' | tail -1)
    echo "[launch] OK: artifacts saved"
    echo "[launch]   final_model.pt        = ${PT_SIZE} bytes"
    echo "[launch]   final_model.int6.ptz  = ${PTZ_SIZE} bytes (alone)"
    if [ -n "$TOTAL_SUB" ]; then
        echo "[launch]   TOTAL submission      = ${TOTAL_SUB} bytes (cap 16,000,000; margin $((16000000 - TOTAL_SUB)))"
        if [ "$TOTAL_SUB" -gt 16000000 ]; then
            echo "[launch] FAIL: total submission OVER CAP — repack via tmp_exec/repack_pergroup.py"
            exit 3
        fi
    fi
    chmod a-w "$PT_PATH" "$PTZ_PATH" 2>/dev/null || true
else
    echo "[launch] FAIL: missing artifacts"
    ls -la "${RUNDIR}/" || true
    exit 2
fi

echo "=========================================================================="
grep -E "stopping_early|diagnostic.*val_bpb|val_loss:|Total submission size|TTT" "${RUNDIR}/train.log" | tail -20 || true
