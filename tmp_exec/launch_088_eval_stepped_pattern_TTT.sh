#!/bin/bash
# Spec 088 — Eval-only matched-compute stepped pattern (073-shape) WITH TTT on, on 060A checkpoint.
#
# Composes 087 (stepped pattern, TTT-off) with TTT — leaderboard-relevant matched-compute stepped-shape test enabled. Real leaderboard-
# relevant test: does the deeper-recurrence gain compose with TTT,
# or does TTT already capture it?
#
# Reuses launch_060_eval.sh (RESUME_FROM_CKPT mode), overrides
# LOOP_PATTERN + leaves TTT/GPTQ ON (vs 080 which disables them).
#
# Outputs to /workspace/runs/083-eval-NL3eq-TTT/seed_42/:
#   train.log           — full pipeline log (TTT phases visible)
#   final_model.int6.ptz — submittable artifact at NL=3 eval
#   final.json          — pre-quant + post-TTT val_bpb numbers
#
# Usage:
#   SEED=42 RUN_LABEL=seed_42 bash tmp_exec/launch_083_eval_NL3eq_TTT.sh
#
# Hardware: 4×H100. Cost ~$3-4, ~25-30 min wall.

set -euo pipefail

SEED="${SEED:?Set SEED}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"

# Pin to e7ccda2 (spec 071's LOOP_PATTERN code; same as 080/081/082).
export SHA="e7ccda2"
export ARM="088-eval-stepped-pattern-TTT"
export RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-/workspace/runs/060A-1855-port/seed_42/final_model.pt}"

# ── Pattern: 4 passes through {3,4,5} = NL=3 equivalent (same as 080) ─────
export LOOP_PATTERN="1,2,3,4,5,3,4,5,4,5,6,6,7,8"
export NUM_LOOPS=2          # any positive value enables looping_active
export ENABLE_LOOPING_AT=0.0  # eval-only — active immediately

# ── Full pipeline: TTT ON, GPTQ ON (key diff from 080) ────────────────────
export TTT_ENABLED=1
export PHASED_TTT_ENABLED=3
export PHASED_TTT_NUM_PHASES=3
export PHASED_TTT_PREFIX_DOCS=2500
export TTT_LORA_RANK=80
export GPTQ_CALIBRATION_BATCHES=16
export GPTQ_RESERVE_SECONDS=4

# Hand off to the standard 060_eval template — it sets all data paths
# and architecture vars from 060A defaults.
exec bash /workspace/parameter-golf/tmp_exec/launch_060_eval.sh
