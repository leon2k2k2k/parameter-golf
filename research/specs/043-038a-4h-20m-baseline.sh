#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a
source research/specs/043-038a-4h-20m-baseline.env
set +a
torchrun --standalone --nproc_per_node=4 \
  records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py
