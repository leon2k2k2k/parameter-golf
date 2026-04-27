#!/bin/bash
# Spec 047D fix3 — Per-pass AdaLN on loop layers (remove allow_in_graph, plain ops).
#
# Fix: removed @torch._dynamo.allow_in_graph from _adabn_apply.
# Prior fix2 (allow_in_graph) blocked gradient flow — γ/β never received gradients.
# Plain elementwise ops (γ*x + β) are natively traceable; gradients flow normally.
#
# Also: ENABLE_LOOPING_AT=0.0 (loop active from step 1 — no mid-training hang).
#
# Compare vs 045 armAC-fix-rerun baseline: pre-quant 1.06479.
# Accept if val_bpb ≤ 1.06479.
set -euo pipefail

SHA="PLACEHOLDER_FIX3_SHA"
ARM="047D-fix3"
RUNDIR="/workspace/runs/047D-loop-adabn-screen/fix3_seed_42"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py"

# ── 1. Git setup ──────────────────────────────────────────────────────────
WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# ── 2. Inductor cache restore ─────────────────────────────────────────────
# fix3 changes only the backward (removes allow_in_graph). Forward graph is same
# as fix2. Restore fix2 stash if available; otherwise seed from 7c6ac51.
SRC="/workspace/.inductor_cache_40a59db_adabn"
SRC_FALLBACK="/workspace/.inductor_cache_7c6ac51"
if [ -d "$SRC" ]; then
  mkdir -p /tmp/inductor_cache
  rsync -a "${SRC}/" /tmp/inductor_cache/
  echo "[launch] cache restored from fix2 stash: $(du -sh /tmp/inductor_cache | cut -f1)"
elif [ -d "$SRC_FALLBACK" ]; then
  mkdir -p /tmp/inductor_cache
  rsync -a "${SRC_FALLBACK}/" /tmp/inductor_cache/
  echo "[launch] cache restored from 7c6ac51 fallback: $(du -sh /tmp/inductor_cache | cut -f1)"
else
  echo "[launch] WARN: no cache stash found — starting cold. Expect ~15 min compile at step 1."
fi
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
export TRITON_AUTOTUNE_NUM_RUNS=1

# ── 3. Deps ───────────────────────────────────────────────────────────────
pip install brotli python-minifier sentencepiece --break-system-packages -q

# ── 4. Output dir ─────────────────────────────────────────────────────────
mkdir -p "$RUNDIR"

# ── 5. Config (identical to 045 armACfix) ─────────────────────────────────
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin'
export VAL_BYTES_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_*.bin'
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 ROPE_YARN=0 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.0
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 TRAIN_LOG_EVERY=100
export ITERATIONS="${ITERATIONS:-20000}"
export WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export VAL_BATCH_TOKENS=524288 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 VAL_LOSS_EVERY=0
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export MATRIX_BITS=6 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0
export EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0
export GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1
export ATTN_OUT_GATE_ENABLED=0 ATTN_OUT_GATE_SRC=proj GATE_WINDOW=12
export RECUR_ALPHA_ENABLED=1 RECUR_DIAG_P2P_COS=0 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export SPINQUANT_ENABLED=0 SPINQUANT_SEED=42 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5 SLOPE_WARMDOWN=-1.0
export SEED=42 MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1200}"
export PHASED_TTT_ENABLED=3 PHASED_TTT_NUM_PHASES=3
export RUN_ID="047D-loop-adabn-screen-fix3"

# ── 6. Inherit 045 AC-fix levers ──────────────────────────────────────────
export LOOP_ITER_EMBEDS=0
export LOOP_SCALE_INIT=recip

# ── 7. 047D lever ─────────────────────────────────────────────────────────
export LOOP_ADABN=1

# ── 8. Skip serialize/quant for the screen ────────────────────────────────
export TRAINING_ONLY_SCREEN=1

# ── 9. Background-rsync stash ─────────────────────────────────────────────
STASH_DST="/workspace/.inductor_cache_${SHA}_adabn_fix3"
mkdir -p "$STASH_DST"
( sleep 600 ; rsync -a /tmp/inductor_cache/ "${STASH_DST}/" ) &
echo "[launch] background stash to ${STASH_DST} scheduled in 10 min"

# ── 10. Train ──────────────────────────────────────────────────────────────
echo "[launch] starting torchrun — worktree: ${WORKTREE}"
echo "[launch] fix3: allow_in_graph REMOVED — γ/β should receive gradients now"
echo "[launch] loop active from step 1 (ENABLE_LOOPING_AT=0.0) — no mid-run hang"
echo "[launch] gate: pre-quant bpb ≤ 1.06479 (accept) | > 1.06679 (kill)"
torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/train.log" 2>&1
