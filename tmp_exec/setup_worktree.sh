#!/bin/bash
# Set up a commit-pinned worktree before launching torchrun.
# Handles all git operations so launch scripts never touch git state during training.
# Idempotent: safe to call with the same args multiple times.
#
# Usage:  WORKTREE=$(bash tmp_exec/setup_worktree.sh <commit-sha> <arm-name>)
# Output: prints the absolute worktree path; all diagnostic messages go to stderr.
#
# Worktree path: /workspace/pg-<arm-name>-<commit-sha>
# Each (arm, commit) pair gets its own directory — safe for concurrent pods on
# the shared /workspace/ volume.
#
# WARNING: git fetch modifies the shared .git/refs. If two pods call this
# simultaneously for commits not yet local, both will fetch concurrently.
# Git uses object-level locks so data corruption is unlikely, but prefer running
# setup for all arms from a single coordinator before launching training pods.

set -euo pipefail

SHA="${1:?Usage: setup_worktree.sh <commit-sha> <arm-name>}"
ARM="${2:?Usage: setup_worktree.sh <commit-sha> <arm-name>}"
WORKTREE="/workspace/pg-${ARM}-${SHA}"
REPO=/workspace/parameter-golf

# Fetch only if the commit isn't known locally — avoids unnecessary shared-.git writes.
if ! git -C "$REPO" cat-file -e "${SHA}^{commit}" 2>/dev/null; then
  echo "[setup] ${SHA} not in local objects — fetching fork..." >&2
  git -C "$REPO" fetch fork >&2
fi

if [ ! -d "$WORKTREE" ]; then
  echo "[setup] creating worktree ${WORKTREE} @ ${SHA}" >&2
  git -C "$REPO" worktree add --detach "$WORKTREE" "$SHA" >&2
else
  echo "[setup] worktree ${WORKTREE} already exists" >&2
fi

# Verify the worktree is at the right commit.
ACTUAL=$(git -C "$WORKTREE" rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$ACTUAL" != "$(git -C "$REPO" rev-parse "$SHA" 2>/dev/null)" ]; then
  echo "[setup] ERROR: worktree HEAD ${ACTUAL} != expected ${SHA}" >&2
  exit 1
fi

echo "$WORKTREE"
