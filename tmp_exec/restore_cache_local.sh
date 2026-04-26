#!/bin/bash
# Restore a stashed inductor cache from the persistent volume to /tmp on the current pod.
# Run ON the pod (not from local machine) before launching torchrun.
#
# Usage:  bash restore_cache_local.sh <commit-sha>
#
# Checks /workspace/.inductor_cache_<sha> on the mounted volume.
# If found: rsync to /tmp/inductor_cache/ (local copy — avoids NFS FUSE "Stale file handle").
# If missing: exits 1 with clear instructions to run prewarm_any.sh first.
#
# After this script succeeds:
#   export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
#   Do NOT set TRITON_AUTOTUNE_NUM_RUNS=1
#   Verify tok/s >= 4,300,000 at step 100

set -euo pipefail
SHA="${1:?Usage: restore_cache_local.sh <commit-sha>}"

SRC="/workspace/.inductor_cache_${SHA}"

if [ ! -d "$SRC" ]; then
  echo "[restore] ERROR: no stashed cache for ${SHA} on this volume."
  echo ""
  echo "[restore] To create it, run on a fresh pod:"
  echo "    bash tmp_exec/prewarm_any.sh ${SHA} <stage1-env> [stage2-env ...]"
  echo ""
  echo "[restore] Then stash from LOCAL MACHINE:"
  echo "    bash tmp_exec/cache_stash.sh <host> <port> ${SHA}"
  exit 1
fi

SIZE=$(du -sh "$SRC" | cut -f1)
echo "[restore] found /workspace/.inductor_cache_${SHA} (${SIZE})"
echo "[restore] rsync to /tmp/inductor_cache/ ..."
mkdir -p /tmp/inductor_cache
rsync -a "${SRC}/" /tmp/inductor_cache/
DEST_SIZE=$(du -sh /tmp/inductor_cache | cut -f1)
echo "[restore] done. /tmp/inductor_cache: ${DEST_SIZE}"
echo "[restore] → export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache"
echo "[restore] → do NOT set TRITON_AUTOTUNE_NUM_RUNS=1"
