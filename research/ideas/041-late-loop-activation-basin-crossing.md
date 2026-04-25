# Idea: Late Loop Activation for Basin Crossing

**Created:** 2026-04-26
**Arc:** 041 family
**Status:** Active — 041Ab and 041Bc running

## Core Hypothesis

The 040 no-loop ablation showed that without recurrence, the loss curve crosses
into a lower basin around step ~5600 (raw train loss 2.29 at step 6100 vs
baseline's 2.31 at step 5100). The baseline loop activates at frac=0.35
(step ~2300), spending ~900 steps in the slow loop phase instead of reaching
that basin.

**Idea:** delay loop activation so the model navigates to the better basin
first in the fast no-loop phase, then activate recurrence to exploit it.
Two gains at once: more total steps (no-loop is faster) and a better starting
basin for the recurrent phase.

## Throughput Model

Step time is linear in total layer passes N:

```
step_time(N) = T_fixed + N × T_layer
             = 22.5ms  + N × 14.5ms
```

Fitted to empirical measurements {N=11 → 182ms, N=15 → 240ms}. Validated at
N=17 (predicted 269ms, actual ~274ms, 2% error).

### Computing N

```
N = 11 + NUM_LOOPS × (LOOP_END - LOOP_START + 1)
```

| Config | N | step_time | tok/s | steps/s |
|--------|---|-----------|-------|---------|
| No loop | 11 | 182ms | 4322K | 5.495 |
| 041Cb (4-5, NL=1) | 13 | 211ms | 3727K | 4.739 |
| 041B/Bc (3-5, NL=1) | 14 | 225ms | 3491K | 4.444 |
| 041A/Ab (4-5, NL=2) | 15 | 240ms | 3277K | 4.167 |
| Baseline (3-5, NL=2) | 17 | 269ms | 2923K | 3.717 |

**Note:** N=13, 14, 15 all appear as 3277K in logs due to 0.1min (6s) time
resolution — 100-step intervals of 21–24s all round to 24s. Use the model for
true throughput; don't trust raw log tok/s for variants within 2 passes of
each other.

**Note:** T_fixed is dominated by fixed per-step overhead (optimizer, gradient
sync, embedding) not layer compute. At 512d/11L on 4×H100, GPU utilization is
~4% — the model is not compute-bound. Looping layers 3-5 vs 4-5 makes no
measurable wall-time difference; only total N matters.

### Frac formula

```
total_steps = frac × 1200 × steps_per_s(11)
            + (1 - frac) × 1200 × steps_per_s(N)

To solve for frac given target T:
frac = (T/1200 - steps_per_s(N)) / (steps_per_s(11) - steps_per_s(N))
```

### Target fracs for ~6100 steps

| N | steps/s (loop) | frac for 6100 |
|---|----------------|---------------|
| 13 | 4.739 | 0.46 |
| 14 | 4.444 | 0.61 |
| 15 | 4.167 | 0.63* |
| 17 | 3.717 | 0.73 |

*041Ab targets ~6000 not 6100; frac=0.63 gives 5989 steps.

## What We've Tested

### 041A — layers 4-5, NUM_LOOPS=2, frac=0.46 (DONE)
- Steps: 5722 (short of target, actual loop 3277K not predicted 3616K)
- val_bpb: **1.06545** (+0.00031 vs baseline 1.06514)
- Result: nearly matched baseline despite late activation and shrunk loop

### 040 — no loop, NUM_LOOPS=0 (DONE)
- Steps: 6569
- val_bpb: **1.07223** (+0.00709 vs baseline)
- Result: recurrence is load-bearing; extra steps alone don't compensate

### 041B — layers 3-5, NUM_LOOPS=1, frac=0.59 (DONE)
- Steps: 6002
- val_bpb: **1.06842** (+0.00328 vs baseline)
- Result: shallow recurrence (1 extra pass) is significantly worse than 2 passes

### 041Ab — layers 4-5, NUM_LOOPS=2, frac=0.63 (RUNNING)
- Target: ~6000 steps
- This is the corrected rerun of 041A at the right frac

### 041Bc — layers 3-5, NUM_LOOPS=1, frac=0.61 (RUNNING)
- Target: ~6100 steps
- Corrected rerun of 041B; expected to show same story (NUM_LOOPS=1 too shallow)

## Key Findings So Far

1. **NUM_LOOPS depth matters more than layer coverage.** 041A (2 passes on 2
   layers) nearly matches baseline. 041B (1 pass on 3 layers) is far worse.
   Extra recurrence depth >> wider recurrence coverage.

2. **Throughput is ~constant for any loop config in this range.** All of N=13,
   14, 15 give effectively the same measured throughput (~3277K) due to log
   resolution. The real differences are 3–7% and only visible in final step
   counts.

3. **Late activation works at frac=0.46 with NUM_LOOPS=2.** Getting ~5722
   steps and +0.00031 vs baseline suggests the basin-crossing idea is real.
   The corrected frac=0.63 (041Ab) should confirm this with ~6000 steps.

## Open Questions

- Does 041Ab (6000 steps, NUM_LOOPS=2) beat baseline? If yes by how much?
- Is there an optimal frac in [0.46, 0.73] for NUM_LOOPS=2?
- Does NUM_LOOPS=3 on fewer layers add value, or is 2 the sweet spot?
- Does the layer range (3-5 vs 4-5) matter at all at NUM_LOOPS=2?
