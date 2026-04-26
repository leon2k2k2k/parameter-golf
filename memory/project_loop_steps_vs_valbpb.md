---
name: Loop steps vs val_bpb relationship and 041 arc lessons
description: Empirical table ~2.2e-6 val_bpb/loop step; three key lessons from 041 arc on depth, frac, and throughput
type: project
---

Empirical finding from the 041 arc (2026-04-26): val_bpb improves roughly linearly with
number of loop steps (NUM_LOOPS=2, layers 4-5 or 3-5, frac varies).

| Spec | Loop steps | val_bpb |
|------|-----------|---------|
| 040 (no loop) | 0 | 1.07223 |
| 041Ab (frac=0.63) | 1834 | 1.06736 |
| 041A (frac=0.46) | 2692 | 1.06545 |
| Baseline (frac=0.35) | 2867 | 1.06514 |

Rate: ~2.2e-6 val_bpb improvement per loop step (in the 1834–2867 range).

Extrapolation check: 1.06736 - (2867-1834) × 2.2e-6 = 1.06509 ≈ baseline 1.06514 ✓

## Three lessons from the 041 arc

**1. Depth > coverage.** NUM_LOOPS=2 (2 extra passes on 2 layers) nearly ties baseline.
NUM_LOOPS=1 (1 extra pass on 3 layers) is +0.003 worse. More recurrence depth per step
beats wider layer coverage.

**2. Too-late activation kills you.** 041Ab proved this: frac=0.63 vs 0.46 gave 260 more
total steps but 858 fewer loop steps — strictly worse (1.06736 vs 1.06545). Delaying to
reach a "better basin" doesn't pay off. The loop steps sacrificed are worth more than the
extra no-loop steps gained.

**3. The real lever is throughput, not frac.** Since frac should stay near 0.46, the only
way to get more loop steps is to make the loop cheaper per step (smaller N). 041D tests
this: 1 loop layer (N=13) instead of 2 (N=15), same NUM_LOOPS=2, same frac=0.46 → +379
loop steps (~3071 vs 2692) → predicted to beat baseline at ~1.06462.

**Why:** Loop steps worth ~2× a no-loop step. Delaying frac sacrifices loop steps at 3:1
ratio to gain total steps — net negative every time.
