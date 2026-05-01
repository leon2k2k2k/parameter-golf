#!/usr/bin/env bash
# Spec 301 — Gated XSA + progressive context on clean HF data
# Base: PR #2014 train_gpt.py + xsa_alpha gate (8 lines); n-gram OFF.
# Usage: SEED=42 bash launch.sh
# Run from any directory — script sets up everything.

set -euo pipefail

SEED="${SEED:-42}"
BRANCH="exp/301-frontier-harvest-clean"
REPO_DIR="/workspace/spec301"
RECORD_DIR="$REPO_DIR/records/track_10min_16mb/2026-05-02_GatedXSA_Progressive3k_ShortDocTTT"
# Tokenizer lives in the #1736 record dir (committed to research branch, inherited here)
TOKENIZER_DIR="$REPO_DIR/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/tokenizers"
ARTIFACT_DIR="/workspace/runs/301-frontier-harvest-clean/seed_${SEED}"
mkdir -p "$ARTIFACT_DIR"

echo "=== Spec 301 launch: seed $SEED ==="
echo "Branch: $BRANCH"
echo "Artifact dir: $ARTIFACT_DIR"

# ── Deps ──────────────────────────────────────────────────────────────────────
pip install brotli huggingface_hub python-minifier sentencepiece --break-system-packages -q

# lrzip required for COMPRESSOR=pergroup; must be present before GPTQ
apt-get install -y lrzip -qq 2>/dev/null
if ! command -v lrzip &>/dev/null; then
    echo "ERROR: lrzip install failed — cannot use pergroup compressor"
    exit 1
fi
echo "[setup] lrzip: $(lrzip --version 2>&1 | head -1)"

# ── Code: clone fork + checkout branch ────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/leon2k2k2k/parameter-golf.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
echo "[setup] branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

# ── Inductor cache restore ─────────────────────────────────────────────────────
if [ -d /workspace/inductor_cache_stash ]; then
    mkdir -p /tmp/inductor_cache
    cp -r /workspace/inductor_cache_stash/. /tmp/inductor_cache/ 2>/dev/null || true
    echo "[setup] inductor cache restored: $(ls /tmp/inductor_cache | wc -l) files"
fi
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache

# ── Data (clean HF dataset) ────────────────────────────────────────────────────
DATA_PATH="/dev/shm/pgolf_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH="$TOKENIZER_DIR/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

if [ ! -d "$DATA_PATH" ]; then
    echo "[data] HF dataset not found — downloading from romeerp/parameter-golf-caseops-v1 ..."
    mkdir -p /dev/shm/pgolf_data
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='/dev/shm/pgolf_data',
    max_workers=8,
)
"
fi

# Verify data
TRAIN_SHARDS=$(ls "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "[data] train_shards: $TRAIN_SHARDS  (expect 80)"
ls "$DATA_PATH"/fineweb_val_*.bin > /dev/null 2>&1 || { echo "ERROR: val shards missing"; exit 1; }

# Verify tokenizer (from the #1736 record dir, committed in our fork)
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: tokenizer not found at $TOKENIZER_PATH"
    exit 1
fi
echo "[data] tokenizer: $TOKENIZER_PATH"

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$RECORD_DIR"

SEED=$SEED \
ARTIFACT_DIR=$ARTIFACT_DIR \
DATA_PATH=$DATA_PATH \
TOKENIZER_PATH=$TOKENIZER_PATH \
\
CASEOPS_ENABLED=1 \
GATED_XSA=1 \
\
NGRAM_TILT_ENABLED=0 \
\
COMPILE_SHAPE_WARMUP=1 \
\
TRAIN_SEQ_SCHEDULE="1024@0.100,2048@0.700,3072@1.000" \
TRAIN_SEQ_LEN=3072 \
EVAL_SEQ_LEN=3072 \
TTT_EVAL_SEQ_LEN=3072 \
EVAL_STRIDE=1536 \
EVAL_INCLUDE_TAIL=1 \
\
PHASED_TTT_NUM_PHASES=1 \
PHASED_TTT_PREFIX_DOCS=2500 \
TTT_SHORT_SCORE_FIRST_ENABLED=1 \
TTT_SHORT_SCORE_FIRST_STEPS="256:8,2000:24" \
TTT_SHORT_DOC_LEN=2000 \
TTT_LORA_RANK=80 \
TTT_MASK=no_qv \
TTT_Q_LORA=0 \
TTT_V_LORA=0 \
TTT_LOCAL_LR_MULT=0.75 \
\
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
MATRIX_LR=0.026 \
QK_GAIN_INIT=5.25 \
GRAD_CLIP_NORM=0.3 \
MIN_LR=0.1 \
GPTQ_RESERVE_SECONDS=4.0 \
\
LQER_RANK=4 \
LQER_TOP_K=3 \
LQER_ASYM_GROUP=64 \
AWQ_LITE_ENABLED=1 \
EMBED_BITS=7 \
COMPRESSOR=pergroup \
\
GATED_ATTN_QUANT_GATE=1 \
ASYM_LOGIT_RESCALE=1 \
SMEAR_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SKIP_GATES_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_SCALE=0.5 \
EMA_DECAY=${EMA_DECAY:-0.998} \
FUSED_CE_ENABLED=1 \
\
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
\
torchrun --nproc_per_node 8 train_gpt.py 2>&1 | tee "$ARTIFACT_DIR/train.log"
