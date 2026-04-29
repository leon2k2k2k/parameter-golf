#!/bin/bash
# Spec 087 — Eval-only matched-compute stepped pattern (073-shape, 2,3,3,2 across {3,4,5,6}) on 060A checkpoint.
#
# Reuses launch_060_eval.sh infrastructure (RESUME_FROM_CKPT mode).
# Overrides LOOP_PATTERN to encode 4 passes through {3,4,5} (vs 060A's
# trained 3 passes), measures pre-quant val_bpb at the new iteration
# count. No new code; uses the LOOP_PATTERN env var from e7ccda2.
#
# Outputs to /workspace/runs/080-eval-deeper-loop/seed_42_NL3eq/:
#   train.log           — full eval log
#   final.json          — bpb measurement
#
# Usage:
#   SEED=42 RUN_LABEL=seed_42_NL3eq bash tmp_exec/launch_080_eval_deeper_loop.sh
#
# Hardware: 4×H100. Cost ~$1, ~10 min wall (compile burst + eval).

set -euo pipefail

SEED="${SEED:?Set SEED}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"

# Pin to e7ccda2 (spec 071's LOOP_PATTERN code; reused verbatim).
export SHA="e7ccda2"
export ARM="087-eval-stepped-pattern"
export RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-/workspace/runs/060A-1855-port/seed_42/final_model.pt}"

# ── Pattern: 4 passes through {3,4,5} = NL=3 equivalent ────────────────────
# body = 1,2,3,4,5,3,4,5,3,4,5,3,4,5,6,7  (16 visits, layers 3-5 each ×4)
# pre [0] + body + post [8,9,10] = 20 total layer-passes (vs 17 for 060A)
export LOOP_PATTERN="1,2,3,4,5,3,4,5,4,5,6,6,7,8"
export NUM_LOOPS=2          # any positive value enables looping_active
export ENABLE_LOOPING_AT=0.0  # eval-only — active immediately on first forward

# ── Diagnostic-only deltas vs 060A defaults ────────────────────────────────
# Skip TTT/GPTQ for the cleanest pre-quant bpb measurement at the new NL.
export TTT_ENABLED=0 PHASED_TTT_ENABLED=0
export GPTQ_CALIBRATION_BATCHES=0 GPTQ_RESERVE_SECONDS=0
export EMA_DECAY=0.0
export SKIP_SUBMISSION_BUILD=1

# Hand off to the standard 060_eval template — it sets all other vars.
exec bash /workspace/parameter-golf/tmp_exec/launch_060_eval.sh
