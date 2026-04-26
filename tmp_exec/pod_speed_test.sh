#!/bin/bash
# Quick pod throughput sanity check. Run after provisioning a fresh pod, BEFORE
# committing to a full training run. Burns ~$0.20-0.40 to confirm we got a fast draw.
#
# Usage:  pod_speed_test.sh <ssh-host> <ssh-port> [commit-sha]
#   commit-sha (optional): if cache is stashed, restore it first. Otherwise compiles cold.
#
# Reference (4×H100, parameter-golf model, pre-loop):
#   - Fast draw:  ~4.30M tok/s
#   - Acceptable: ≥ 4.085M tok/s (≤ 5% below ref)
#   - Slow:       < 4.085M  → reprovision
#
# Strategy: run a 60-step micro-train on the actual repo + train_gpt.py, kill it after
# step 60 lands, parse the tok/s reading. Total time ~3-4 min including compile.

set -euo pipefail
HOST="$1"; PORT="$2"; SHA="${3:-}"
SSH="ssh -o StrictHostKeyChecking=no -i $HOME/.runpod/ssh/RunPod-Key-Go root@$HOST -p $PORT"

# Restore cache if commit-sha provided + stash exists
if [ -n "${SHA}" ]; then
  if $SSH "[ -d /workspace/.inductor_cache_${SHA} ]"; then
    echo "[speed] restoring stashed cache for ${SHA} (~30s)"
    $SSH "mkdir -p /tmp/inductor_cache && rsync -a /workspace/.inductor_cache_${SHA}/ /tmp/inductor_cache/"
  else
    echo "[speed] no stash for ${SHA} on volume; compiling cold this run"
  fi
fi

$SSH 'pip install brotli sentencepiece --break-system-packages -q' >/dev/null 2>&1

# Launch a micro-run with TRAIN_LOG_EVERY=10 ITERATIONS=80 MAX_WALLCLOCK_SECONDS=240.
# Use the same train_gpt.py shape as the real run so the warmup throughput is comparable.
cat > /tmp/_speed_launch.sh <<EOS
#!/bin/bash
set -e
# Use a commit-specific worktree so concurrent pods don't clobber each other.
# /workspace/parameter-golf HEAD can be on any commit — never use it directly.
if [ -z "${SHA}" ]; then
  echo "[speed] ERROR: commit-sha arg required. Usage: pod_speed_test.sh <host> <port> <sha>"
  echo "[speed] Using /workspace/parameter-golf directly is unsafe with multiple pods sharing"
  echo "[speed] the same volume — another pod's git checkout can silently switch the directory."
  exit 1
fi
WORKTREE="/workspace/pg-speed-${SHA}"
if [ ! -d "\$WORKTREE" ]; then
  git -C /workspace/parameter-golf worktree add --detach "\$WORKTREE" "${SHA}"
fi
cd "\$WORKTREE"
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
# TRITON_AUTOTUNE_NUM_RUNS=1 intentionally NOT set: it costs ~6% throughput at 4xH100
# (picks first-candidate kernels). Without cache, Stage 1 compile is slower but tok/s
# reading reflects actual training throughput. With cache, optimal kernels load directly.
export DATA_DIR=/workspace/parameter-golf/data
export DATASETS_DIR='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved'
export TOKENIZER_PATH='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model'
export TRAIN_FILES='/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin'
export VAL_FILES="${TRAIN_FILES}" VAL_BYTES_FILES="${TRAIN_FILES}"
export VOCAB_SIZE=8192 NUM_LAYERS=11 XSA_LAST_N=11 MODEL_DIM=512 NUM_KV_HEADS=4 NUM_HEADS=8
export MLP_MULT=4 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30 ROPE_BASE=10000 ROPE_DIMS=16
export ROPE_TRAIN_SEQ_LEN=2048 LN_SCALE=1 QK_GAIN_INIT=5.0
export NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=999.0  # never activate loop in speed test
export PARALLEL_START_LAYER=8 PARALLEL_FINAL_LANE=mean
export MIN_LR=0.1 EMBED_LR=0.6 TIED_EMBED_LR=0.03 TIED_EMBED_INIT_STD=0.005
export MATRIX_LR=0.026 SCALAR_LR=0.02 MUON_MOMENTUM=0.97 MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1
export BETA1=0.9 BETA2=0.95 ADAM_EPS=1e-8 GRAD_CLIP_NORM=0.3 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085
export EMA_DECAY=0.9965 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
export TRAIN_LOG_EVERY=10 ITERATIONS=80 WARMDOWN_FRAC=0.75 WARMUP_STEPS=20
export MAX_WALLCLOCK_SECONDS=300 VAL_LOSS_EVERY=0 TTT_ENABLED=0 TRAINING_ONLY_SCREEN=1
export CASEOPS_ENABLED=1 COMPRESSOR=brotli
export MATRIX_BITS=6 EMBED_BITS=7
export SKIP_GATES_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1
export RECUR_ALPHA_ENABLED=1 SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
export MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5
export SEED=42 RUN_ID="podspeed"
torchrun --standalone --nproc_per_node=4 "\$WORKTREE/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py" >/tmp/podspeed.log 2>&1
EOS
scp -o StrictHostKeyChecking=no -i "$HOME/.runpod/ssh/RunPod-Key-Go" -P "$PORT" /tmp/_speed_launch.sh "root@${HOST}:/tmp/_speed_launch.sh" >/dev/null
$SSH 'mkdir -p /workspace/runs/_podspeed && setsid bash /tmp/_speed_launch.sh </dev/null >/tmp/podspeed.out 2>&1 & disown'

# Poll until step 60 lands or 5 min timeout
echo "[speed] launched. Waiting for step 60..."
for i in $(seq 1 60); do
  sleep 10
  STEP_LINE=$($SSH 'grep -E "^[0-9]+/[0-9]+ train_loss:" /tmp/podspeed.log 2>/dev/null | tail -1' || true)
  if [ -n "$STEP_LINE" ]; then
    STEP=$(echo "$STEP_LINE" | awk -F/ '{print $1}')
    if [ "$STEP" -ge 60 ]; then
      TOKS=$(echo "$STEP_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="tok/s:") print $(i+1)}')
      echo "[speed] step ${STEP}: tok/s = ${TOKS}"
      RATIO=$(awk -v t=$TOKS 'BEGIN{printf "%.3f", t/4300000.0}')
      echo "[speed] ratio vs reference 4.30M: ${RATIO}"
      VERDICT=$(awk -v t=$TOKS 'BEGIN{if (t>=4085000) print "OK"; else print "SLOW (reprovision)"}')
      echo "[speed] VERDICT: ${VERDICT}"
      $SSH 'pkill -9 -f "train_gpt.py" 2>/dev/null; pkill -9 -f torchrun 2>/dev/null; true' >/dev/null 2>&1
      exit 0
    fi
  fi
done
echo "[speed] TIMED OUT — step 60 did not land in 10 min"
$SSH 'pkill -9 -f "train_gpt.py" 2>/dev/null; pkill -9 -f torchrun 2>/dev/null; true' >/dev/null 2>&1
exit 2
