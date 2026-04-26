---
name: Never allow recompile during training runs
description: Every new commit must have its inductor cache pre-warmed before any real training run. Recompile during training is never acceptable.
type: feedback
---

Recompile during training is NEVER allowed. A cold inductor cache mid-run eats into the 20-min wallclock budget, causes step-count mismatch vs baseline, and makes val_bpb comparisons unfair.

**Why:** AC-fix (fc54262) was the first pod ever on that commit — compiled from cold mid-training at loop activation, cache grew from 1.7GB→2.0GB during the run. Old arms (1c6cd7c) all had warm caches from prior runs so showed no pause. The comparison is biased.

**How to apply:** Every time a new branch/commit is created for an exp arm:
1. Before any real training run, do a **prewarm run**: launch training for ~100 steps, then kill it. This populates the inductor cache on the pod.
2. Then launch the real training run — cache is warm, no recompile penalty.
3. All arms on the SAME commit can share the same pod's cache (run sequentially on same pod).
4. If switching commits mid-pod, prewarm again.

The project already has `runs/042B-prewarm-speedup-smoke` which explored this — reference it.

In specs: add a **prewarm step** to the execution checklist for any arm on a new commit. It costs ~2 min and saves the entire compile budget.
