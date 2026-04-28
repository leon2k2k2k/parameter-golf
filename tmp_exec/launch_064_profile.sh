#!/bin/bash
# Spec 064 — Bottleneck profiling diagnostic on 060A.
#
# Runs 060A's pinned commit (a0a48b7) verbatim under nsys profile,
# capturing CUDA / NVTX / OSRT / cuDNN / cuBLAS traces with GPU
# metrics. No model changes. Diagnostic-only run: TTT/EMA/GPTQ
# disabled to keep the profiled window short and focused on training.
#
# Outputs to /workspace/runs/064-profile-bottleneck/seed_42/:
#   profile_full.nsys-rep   — full Nsight Systems trace
#   profile_full.sqlite     — exported queryable form
#   train.log               — full stdout/stderr
#   kernel_summary_pre_loop.txt   (post-run, by analyze_064_profile.py)
#   kernel_summary_post_loop.txt
#   vram_curve.csv
#   toks_per_s.csv
#
# Usage:
#   SEED=42 RUN_LABEL=seed_42 bash tmp_exec/launch_064_profile.sh
#
# Hardware: 4×H100. Cost ~$1.50, ~17 min wall.

set -euo pipefail

SEED="${SEED:?Set SEED}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"

# Pin to 060A baseline commit (head of exp/060-resume-ckpt as of 2026-04-29).
SHA="a0a48b7"
ARM="064-profile-bottleneck"
RUNDIR="/workspace/runs/${ARM}/${RUN_LABEL}"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py"

WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# System deps
which lrzip > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y lrzip) 2>&1 | tail -5
pip install brotli python-minifier sentencepiece --break-system-packages -q

# nsys preflight — bail clearly if absent so execution can install
if ! command -v nsys > /dev/null 2>&1; then
  echo "[launch_064] nsys (Nsight Systems CLI) NOT FOUND on PATH" >&2
  echo "[launch_064] try: apt-get install -y nvidia-nsight-systems-cli" >&2
  echo "[launch_064] OR:  pip install nvidia-nsight" >&2
  exit 2
fi
echo "[launch_064] nsys: $(nsys --version | head -1)"

mkdir -p "$RUNDIR"
cd "$RUNDIR"

# Inductor cache: must be on /tmp (memory: feedback_inductor_cache_on_tmp).
# Cold compile is fine for a 250-step diagnostic — autotune kicks in early
# and we don't need a fully warmed cache to read kernel rankings.
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
mkdir -p /tmp/inductor_cache
bash /workspace/parameter-golf/tmp_exec/restore_cache_local.sh "$SHA" 2>/dev/null \
  && echo "[launch_064] cache restored" || echo "[launch_064] cold compile (ok for diagnostic)"

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
# Goal: cover ~250 steps spanning the loop activation event (35% of 250 =
# step ~88), with all eval-side machinery disabled.
export ITERATIONS=250
export MAX_WALLCLOCK_SECONDS=240
export WARMDOWN_FRAC=0.85 WARMUP_STEPS=20
export GRAD_ACCUM_STEPS=2          # 4H matched-FLOPs to 8H × 600s
export EMA_DECAY=0.0               # disable EMA (saves end-of-step time)
export TTT_ENABLED=0 PHASED_TTT_ENABLED=0   # disable TTT eval (~min savings)
# Quant-side: leave LQER + bit configs in place so nothing crashes during
# init/save, but don't actually serialize a submission blob.
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=14.0 GPTQ_CALIBRATION_BATCHES=0 GPTQ_RESERVE_SECONDS=0
export TTT_LORA_RANK=80

# Skip submission build entirely.
export SKIP_SUBMISSION_BUILD=1

export SEED="$SEED"
export RUN_ID="${ARM}-${RUN_LABEL}"
export ARTIFACT_DIR="$RUNDIR"

echo "=========================================================================="
echo "[launch_064] DIAGNOSTIC RUN — 060A @ ${SHA} under nsys"
echo "[launch_064] SEED=${SEED} / RUN_LABEL=${RUN_LABEL}"
echo "[launch_064] ITERATIONS=250 WALLCLOCK=240s GRAD_ACCUM=2 (4H)"
echo "[launch_064] OUTPUT=${RUNDIR}"
echo "=========================================================================="

# nsys captures the full timeline. We slice into pre/post-loop windows in
# the analysis script (loop activation step is in train.log).
nsys profile \
  --output="${RUNDIR}/profile_full" \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=cpu \
  --gpu-metrics-device=all \
  --force-overwrite=true \
  --capture-range=none \
  torchrun --standalone --nproc_per_node=4 \
    "${WORKTREE}/${TRAIN_SCRIPT}" \
  2>&1 | tee "${RUNDIR}/train.log"

# Export sqlite for queryability (nsys CLI can do this directly).
echo "[launch_064] exporting sqlite..."
nsys export --type=sqlite --output="${RUNDIR}/profile_full.sqlite" \
  --force-overwrite=true "${RUNDIR}/profile_full.nsys-rep"

# Post-run analysis: kernel rankings in pre/post-loop windows + CSV scrapes.
echo "[launch_064] running analyze_064_profile.py..."
python3 /workspace/parameter-golf/tmp_exec/analyze_064_profile.py \
  --sqlite "${RUNDIR}/profile_full.sqlite" \
  --train-log "${RUNDIR}/train.log" \
  --out-dir "${RUNDIR}" \
  --enable-looping-frac 0.35 \
  --total-iterations 250

echo "=========================================================================="
echo "[launch_064] DONE. Inspect:"
echo "[launch_064]   ${RUNDIR}/kernel_summary_pre_loop.txt"
echo "[launch_064]   ${RUNDIR}/kernel_summary_post_loop.txt"
echo "[launch_064]   ${RUNDIR}/vram_curve.csv"
echo "[launch_064]   ${RUNDIR}/toks_per_s.csv"
echo "[launch_064]   ${RUNDIR}/profile_full.nsys-rep   (open in Nsight Systems GUI)"
echo "=========================================================================="
