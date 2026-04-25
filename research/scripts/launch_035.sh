#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
record_dir="/workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT"
artifact_dir="/workspace/runs/035-min-lr-on-030-family/run_a/seed_314"

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  data_dir="/workspace"
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  data_dir="/workspace/parameter-golf/data"
else
  echo "CaseOps tokenizer not found under either JP or NA layout" >&2
  exit 1
fi

mkdir -p /workspace/.torch_inductor_cache "$artifact_dir"

export NCCL_NET=Socket
export DATA_DIR="$data_dir"
export ARTIFACT_DIR="$artifact_dir"
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache
export RUN_ID=035A_seed314
export CASEOPS_ENABLED=1
export TTT_ENABLED=0
export MLP_CLIP_SIGMAS=12.0
export ATTN_CLIP_SIGMAS=13.0
export EMBED_BITS=7
export EMBED_CLIP_SIGMAS=15.0
export MATRIX_LR=0.026
export GATED_ATTN_ENABLED=1
export GATED_ATTN_INIT_STD=0.005
export GATED_ATTN_QUANT_GATE=1
export RECUR_ALPHA_ENABLED=1
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export GPTQ_RESERVE_SECONDS=4
export GPTQ_CALIBRATION_BATCHES=16
export MIN_LR=0.10
export MAX_WALLCLOCK_SECONDS=1200
export TRAIN_LOG_EVERY=100
export SEED=314

cd "$record_dir"
git fetch fork
git checkout c3a99b3
python3 /tmp/verify_035_inheritance.py \
  --record-dir "$record_dir" \
  --artifact-dir "$artifact_dir" \
  --expected-min-lr 0.10 \
  --label 035A
exec torchrun --standalone --nproc_per_node=4 train_gpt.py
