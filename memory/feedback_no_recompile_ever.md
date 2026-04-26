---
name: Never allow recompile during training runs
description: Every new commit must have its inductor cache pre-warmed before any real training run. Recompile during training is never acceptable.
type: feedback
---

Recompile during training is NEVER allowed. On multi-GPU runs it doesn't just bias step counts — it **kills the run**. Mid-training recompile causes ranks to desynchronize NCCL collectives (different compile timing per rank → different op order/shape → collective hang → NCCL timeout → dead job). AC-fix arm died exactly this way: rank 3 had 4 outstanding NCCL ops that never completed.

**Why:** AC-fix (fc54262) was the first pod ever on that commit — compiled from cold mid-training at loop activation, cache grew from 1.7GB→2.0GB during the run. Old arms (1c6cd7c) all had warm caches from prior runs so showed no pause. The comparison is biased.

**Root cause:** The training code has a built-in precompile bracket (`slope_anneal: precompiled (forward+forward_logits, both looping states)`) during warmup that explicitly compiles BOTH the pre-loop and loop-active graphs. When cache is warm (prior run on same commit), this completes instantly → 0 mid-training recompiles. When cache is cold (new commit, new pod), it takes ~15 min without tweaks, ~8 min with tweaks — if it spills into training time, you get mid-training recompiles.

**During every run — actively monitor for mid-training recompiles:**
Watch the train.log for `Recompiling` or `guard failed` lines AFTER the precompile bracket finishes (`slope_anneal: precompiled`). Any recompile after that line is wrong and the run must be killed immediately — do not let it continue. On 4×H100, mid-training recompile will cause NCCL collective desync and a full hang within minutes.

**How to apply:** For any arm on a new commit, ALWAYS set:
```
TORCHINDUCTOR_COMPILE_THREADS=8
TRITON_AUTOTUNE_NUM_RUNS=1
```
These cut cold compile from ~15 min to ~8 min (discovered in `runs/042B-prewarm-speedup-smoke`). The precompile bracket then completes before real training steps begin.

All arms on the SAME commit on the SAME pod share the cache — only the first arm pays the compile cost. Sequential arms on same pod are free.
