#!/bin/bash
# Stash a pod's hot inductor cache to the persistent volume so a future pod
# can hot-restore (skip torch.compile + Triton autotune entirely).
#
# Usage:  cache_stash.sh <ssh-host> <ssh-port> <commit-sha>
#   ssh-host: pod IP from `runpodctl ssh info`
#   ssh-port: pod SSH port
#   commit-sha: the train_gpt.py commit whose graph this cache encodes (e.g. fc54262)
#
# Stashed to /workspace/.inductor_cache_<commit-sha>/. ~2 GB, ~30s rsync.
# Cache is graph-hash keyed → reusable on any pod with same GPU SKU (H100 → H100 OK).

set -euo pipefail
HOST="$1"; PORT="$2"; SHA="$3"
SSH="ssh -o StrictHostKeyChecking=no -i $HOME/.runpod/ssh/RunPod-Key-Go root@$HOST -p $PORT"

DEST="/workspace/.inductor_cache_${SHA}"
echo "[stash] rsync /tmp/inductor_cache/ → ${DEST}/  (volume — survives pod stop)"
$SSH "mkdir -p '${DEST}' && rsync -a --delete /tmp/inductor_cache/ '${DEST}/' && du -sh '${DEST}'"
echo "[stash] done. Future pods can call: cache_restore.sh <host> <port> ${SHA}"
