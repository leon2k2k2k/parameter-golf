#!/bin/bash
# Spec 064b — torch.profiler diagnostic on 060A baseline.
#
# Pivot from spec 064 after container's bundled nsys lacked CUPTI
# kernel tracing. Uses torch.profiler instead — in-process, no privilege
# requirements. Captures two chrome traces (pre and post loop activation).
#
# Outputs to /workspace/runs/064b-torch-profile/seed_42/:
#   chrome_trace_step_<N>.json  — torch.profiler chrome traces (rank 0 only)
#   train.log                   — full stdout/stderr
#
# Usage:
#   SEED=42 RUN_LABEL=seed_42 bash tmp_exec/launch_064b_torch_profile.sh
#
# Hardware: 4×H100. Cost ~$2-3, ~10-15 min wall.

set -euo pipefail

SEED="${SEED:?Set SEED}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"

# Pin to 064b (060A baseline + torch.profiler patch).
SHA="2db4d41"
ARM="064b-torch-profile"
RUNDIR="/workspace/runs/${ARM}/${RUN_LABEL}"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py"

WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# System deps
which lrzip > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y lrzip) 2>&1 | tail -5
pip install brotli python-minifier sentencepiece --break-system-packages -q

mkdir -p "$RUNDIR"
cd "$RUNDIR"

export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache
bash /workspace/parameter-golf/tmp_exec/restore_cache_local.sh "$SHA" 2>/dev/null \
  && echo "[launch_064b] cache restored" || echo "[launch_064b] cold compile (ok for diagnostic)"

# ── Data paths (verbatim 060A) ────────────────────────────────────────────
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'

# ── Architecture (verbatim 060A) ──────────────────────────────────────────
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean

# ── Optimizer (verbatim 060A) ─────────────────────────────────────────────
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.99 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085

# ── Data + scheduling ─────────────────────────────────────────────────────
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=10
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
export CASEOPS_ENABLED=1 COMPRESSOR=pergroup
export SMEAR_GATE_ENABLED=1 SKIP_GATES_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=0.5
export GATED_ATTN_ENABLED=0 ATTN_OUT_GATE_ENABLED=0

# ── Diagnostic-only deltas vs 060A ────────────────────────────────────────
export ITERATIONS=250
export MAX_WALLCLOCK_SECONDS=240
export WARMDOWN_FRAC=0.85 WARMUP_STEPS=20
export GRAD_ACCUM_STEPS=2          # 4H matched-FLOPs to 8H × 600s
export EMA_DECAY=0.0
export TTT_ENABLED=0 PHASED_TTT_ENABLED=0
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=14.0 GPTQ_CALIBRATION_BATCHES=0 GPTQ_RESERVE_SECONDS=0
export TTT_LORA_RANK=80
export SKIP_SUBMISSION_BUILD=1

# ── torch.profiler harness ────────────────────────────────────────────────
export PROFILE_CHROME_TRACE=1
export PROFILE_OUTPUT_DIR="$RUNDIR"
export PROFILE_SKIP_FIRST=78
export PROFILE_ACTIVE_STEPS=8
export PROFILE_REPEAT=2

export SEED="$SEED"
export RUN_ID="${ARM}-${RUN_LABEL}"
export ARTIFACT_DIR="$RUNDIR"

echo "=========================================================================="
echo "[launch_064b] DIAGNOSTIC RUN — 060A+profiler @ ${SHA}"
echo "[launch_064b] SEED=${SEED} / RUN_LABEL=${RUN_LABEL}"
echo "[launch_064b] ITERATIONS=250 WALLCLOCK=240s GRAD_ACCUM=2 (4H)"
echo "[launch_064b] PROFILE windows: skip_first=78 active=8 repeat=2"
echo "[launch_064b]   → window 1 ≈ steps 80-87 (pre-loop)"
echo "[launch_064b]   → window 2 ≈ steps 90-97 (post-loop, loop activates ~88)"
echo "[launch_064b] OUTPUT=${RUNDIR}"
echo "=========================================================================="

torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  2>&1 | tee "${RUNDIR}/train.log"

echo "=========================================================================="
echo "[launch_064b] DONE. Inspect:"
ls -la "${RUNDIR}"/chrome_trace_*.json 2>/dev/null || echo "  (no chrome traces written — check train.log for profiler:* lines)"
echo "=========================================================================="
