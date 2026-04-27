#!/bin/bash
# Spec 047C smoke — 200-step quick verification before committing to full run.
#
# Goals:
#   - No NaN / divergence
#   - step_time within 10% of baseline (~4.30M tok/s on 4×H100)
#   - Loss at step 0 matches AC-fix-rerun within rounding (LoRA delta = 0 at init)
#   - LoRA grads non-zero (loop is active from step 1 — verify with grep "loop_ffn" train.log)
#
# Cost: ~3 min, ~$0.30. Hardware: 4×H100, NE-1.
#
# ENABLE_LOOPING_AT=0.0: loop is active from step 1. No mid-training transition.
# This eliminates the hang at loop activation that plagued earlier attempts.
# The main script now defaults to 0.0, so no override needed here.
set -euo pipefail

export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=180
exec bash "$(dirname "$0")/launch_047C.sh"
