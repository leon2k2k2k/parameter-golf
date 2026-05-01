#!/usr/bin/env bash
# Spec 302 — corrected clean #2118 recipe launch
# Run from: /workspace/2118/records/track_10min_16mb/2026-05-01_Gated XSA + token-only n-gram TTT + GPTQ_RESERVE=2.0/
# Usage: SEED=42 bash launch.sh

set -euo pipefail

SEED="${SEED:-42}"
RECORD_DIR="/workspace/2118/records/track_10min_16mb/2026-05-01_Gated XSA + token-only n-gram TTT + GPTQ_RESERVE=2.0"
ARTIFACT_DIR="/workspace/runs/302-clean-2118-recipe/seed_${SEED}"
mkdir -p "$ARTIFACT_DIR"

echo "=== Spec 302 launch: seed $SEED ==="
echo "Artifact dir: $ARTIFACT_DIR"

# ── Data (HF download, confirmed clean) ───────────────────────────────────────
DATA_PATH="/dev/shm/pgolf_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH="${RECORD_DIR}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

# Verify HF data is present
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: HF data not found at $DATA_PATH — download first"
    exit 1
fi

TRAIN_SHARDS=$(ls "$DATA_PATH"/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "train_shards: $TRAIN_SHARDS  (expect 80)"
ls "$DATA_PATH"/fineweb_val_*.bin > /dev/null 2>&1 || { echo "ERROR: val shards missing"; exit 1; }

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$RECORD_DIR"

SEED=$SEED \
ARTIFACT_DIR=$ARTIFACT_DIR \
DATA_PATH=$DATA_PATH \
TOKENIZER_PATH=$TOKENIZER_PATH \
\
CASEOPS_ENABLED=1 \
GATED_XSA_ENABLED=1 \
\
NGRAM_TILT_ENABLED=0 \
\
MIN_LR=0.1 \
EMBED_BITS=7 \
EVAL_SEQ_LEN=2560 \
TTT_EVAL_SEQ_LEN=2560 \
TTT_LORA_RANK=80 \
PHASED_TTT_PREFIX_DOCS=1000 \
PHASED_TTT_NUM_PHASES=1 \
GPTQ_RESERVE_SECONDS=2.0 \
\
SKYLIGHT_MUON=0 \
\
AWQ_LITE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
COMPRESSOR=pergroup \
\
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
\
torchrun --nproc_per_node 8 train_gpt.py 2>&1 | tee "$ARTIFACT_DIR/train.log"
