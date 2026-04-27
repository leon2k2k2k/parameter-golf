#!/bin/bash
# Spec 047C smoke — 200-step quick verification before committing to full run.
#
# Goals:
#   - No NaN / divergence
#   - step_time pre-loop within 10% of baseline (~4.30M tok/s on 4×H100)
#   - Loss at step 0 matches AC-fix-rerun within rounding (LoRA delta = 0 init)
#   - LoRA grads non-zero post-loop-activation (verify with grep "loop_ffn" train.log)
#
# Cost: ~3 min, ~$0.30. Hardware: 4×H100, NE-1.
#
# ENABLE_LOOPING_AT=0.05: loop fires at 0.05 * 180s = 9s (~step 45), so the
# full 200 steps exercise the LoRA forward path. Default 0.35 would fire at 63s
# but pre-loop step rate (~5.5 steps/s) exhausts 200 steps in 36s — before loop.
set -euo pipefail

export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=180
export ENABLE_LOOPING_AT=0.05
exec bash "$(dirname "$0")/launch_047C.sh"
