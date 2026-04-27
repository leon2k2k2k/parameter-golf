#!/bin/bash
# Spec 047C smoke — 200-step quick verification before committing to full run.
#
# Goals:
#   - No NaN / divergence
#   - step_time pre-loop within 10% of baseline (~4.30M tok/s on 4×H100)
#   - Loss at step 0 matches AC-fix-rerun within rounding (LoRA delta = 0 init)
#   - LoRA grads non-zero post-loop-activation (manually inspect log if doubting)
#
# Cost: ~3 min, ~$0.30. Hardware: 4×H100, NE-1.
#
# This is launch_047C.sh with ITERATIONS=200 and MAX_WALLCLOCK_SECONDS=180.
# Note: looping activates at frac=0.35 of MAX_WALLCLOCK_SECONDS, so with
# MWS=180 the loop fires around 60s in, well before the 200-step cap. That
# means the smoke DOES exercise the LoRA forward path (the whole point).
set -euo pipefail

export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=180
exec bash "$(dirname "$0")/launch_047C.sh"
