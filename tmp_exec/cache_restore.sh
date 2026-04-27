#!/bin/bash
# Restore a stashed inductor cache from the persistent volume to /tmp on a fresh pod.
# Run BEFORE launching torchrun. After this, set TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache.
#
# Usage:  cache_restore.sh <ssh-host> <ssh-port> <commit-sha>
#
# Reminder per memory: TORCHINDUCTOR_CACHE_DIR must point at /tmp, NOT the volume directly
# (NFS FUSE causes Triton "Stale file handle" deaths). This script only copies, it does
# not redirect the cache dir.

set -euo pipefail
HOST="$1"; PORT="$2"; SHA="$3"
SSH="ssh -o StrictHostKeyChecking=no -i $HOME/.runpod/ssh/RunPod-Key-Go root@$HOST -p $PORT"

SRC="/workspace/.inductor_cache_${SHA}"
echo "[restore] checking volume for ${SRC}"
EXISTS=$($SSH "[ -d '${SRC}' ] && du -sh '${SRC}' || echo MISSING")
if echo "$EXISTS" | grep -q MISSING; then
  echo "[restore] no stashed cache for ${SHA} on this volume — pod will compile cold"
  exit 1
fi
echo "[restore] found: ${EXISTS}"
echo "[restore] rsync ${SRC}/ → /tmp/inductor_cache/"
$SSH "mkdir -p /tmp/inductor_cache && rsync -a '${SRC}/' /tmp/inductor_cache/ && du -sh /tmp/inductor_cache"
echo "[restore] done. Set TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache in launch envs."
