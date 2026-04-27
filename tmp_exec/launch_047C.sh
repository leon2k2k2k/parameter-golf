#!/bin/bash
# Spec 047C — Per-pass FFN LoRA on looped layers (rank-2, both up + down).
# Mirrors launch_045_armACfix.sh + LOOP_FFN_LORA_RANK=2 + TRAINING_ONLY_SCREEN=1.
#
# Region: NE-1 preferred (per memory 2026-04-27). JP volume ID is fallback.
# Hardware: 4×H100.
#
# Compile concerns (per [NEVER tolerate mid-run torch.compile recompile]):
#   - LoRA forward introduces new matmul (A@B) + addition + fp32→bf16 cast inside
#     the existing fused MLP path. The loop-active LoRA graph is COLD.
#   - We restore the fc54262 cache (which has the loop-active non-LoRA graph hot)
#     and accept TRITON_AUTOTUNE_NUM_RUNS=1 to bound autotune cost on cold paths.
#   - Expected: a small (10–30s) compile pause when looping flips on at frac=0.35
#     (~step 1750). KILL if step_time > 5× baseline for >10 steps after step 1750.
#   - Proper prewarm (prewarm_5cf60f9.sh) is for the multi-seed promotion path,
#     not the screen.
#
# Smoke variant: set ITERATIONS=200 MAX_WALLCLOCK_SECONDS=180 in the env before
# invoking this script, OR see launch_047C_smoke.sh.
set -euo pipefail

SHA="0826944"
ARM="047C"
RUNDIR="/workspace/runs/047C-per-pass-lora-ffn/seed_42"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py"

# ── 1. Git setup (all git ops here — none after this block) ───────────────
WORKTREE=$(bash /workspace/parameter-golf/tmp_exec/setup_worktree.sh "$SHA" "$ARM")

# ── 2. Inductor cache restore ─────────────────────────────────────────────
# Use fc54262 cache: covers the loop-active non-LoRA graph. The new LoRA
# subgraph (matmul A@B, addition, fp32→bf16 cast) will autotune fresh with
# TRITON_AUTOTUNE_NUM_RUNS=1 below — bounded cost for screen.
SRC="/workspace/.inductor_cache_fc54262"
if [ ! -d "$SRC" ]; then
  echo "[launch] WARN: no fc54262 cache stashed — starting cold."
  echo "[launch] First training step will be slow but should converge."
else
  mkdir -p /tmp/inductor_cache
  rsync -a "${SRC}/" /tmp/inductor_cache/
  echo "[launch] cache restored from fc54262: $(du -sh /tmp/inductor_cache | cut -f1)"
fi
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
# 047C: LoRA paths are cold even with fc54262 cache — bound autotune cost.
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
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT="${ENABLE_LOOPING_AT:-0.35}"
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
export RUN_ID="047C-per-pass-lora-ffn"

# ── 6. Inherit 045 AC-fix levers (baseline being compared against) ────────
export LOOP_ITER_EMBEDS=1
export LOOP_SCALE_INIT=recip

# ── 7. 047C lever ─────────────────────────────────────────────────────────
export LOOP_FFN_LORA_RANK=2

# ── 8. Skip serialize/quant for the screen ────────────────────────────────
# Per [Screen via training-endpoint val_bpb] memory — saves ~$3-4, gives the
# pre-quant post-EMA val_bpb signal we want. Quant wiring is out of scope for
# this commit (LoRA banks would need passthrough fp16 or LQER int4 handling).
export TRAINING_ONLY_SCREEN=1

# ── 9. Background-rsync stash on launch (per memory) ──────────────────────
# Stash the inductor cache to volume IN PARALLEL with training, so future
# arms can restore without a fresh prewarm. Safe: rsync of /tmp/* doesn't
# block training, and the cache is final by the time the rsync completes.
STASH_DST="/workspace/.inductor_cache_${SHA}"
mkdir -p "$STASH_DST"
( sleep 600 ; rsync -a /tmp/inductor_cache/ "${STASH_DST}/" ) &
echo "[launch] background stash to ${STASH_DST} scheduled in 10 min"

# ── 10. Train (no git ops below this line) ────────────────────────────────
echo "[launch] starting torchrun — worktree: ${WORKTREE}"
echo "[launch] verify tok/s >= 4,300,000 at step 500 before committing to full run"
echo "[launch] WATCH: step_time spike at frac=0.35 (~step 1750) is expected;"
echo "[launch] KILL if step_time > 5× baseline for >10 steps post-spike"
torchrun --standalone --nproc_per_node=4 \
  "${WORKTREE}/${TRAIN_SCRIPT}" \
  >> "${RUNDIR}/train.log" 2>&1
