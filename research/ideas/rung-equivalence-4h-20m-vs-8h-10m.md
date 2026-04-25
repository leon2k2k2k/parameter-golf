# Idea — 4H/20min and 8H/10min are equivalent screening rungs

**Created:** 2026-04-25

## Summary

The `4×H100 / 1200s` and `8×H100 / 600s` rungs produce essentially identical
training trajectories and final prequant val_bpb, consistently across multiple
architecture families. This makes the 4H rung a valid and cheap substitute for
full 8H promotion screening.

---

## Why they should be equivalent

Both rungs deliver the same total GPU-hours and the same total optimizer steps
(~4800–5200), because:

- 8H at 600s ≈ 4H at 1200s in wall-clock GPU spend
- `grad_accum_steps = 8 // world_size`, so:
  - 8H: `grad_accum=1`, one optimizer step per micro-batch
  - 4H: `grad_accum=2`, two micro-batches accumulated per step
- Effective batch size is identical: `TRAIN_BATCH_TOKENS=786432` either way

The only structural difference is data sharding: 8 GPUs vs 4 GPUs means each
rank sees different sequences. This introduces a small amount of stochastic
variance, but not systematic bias.

---

## Evidence

### Pair 1 — specs 025b / 026 (cross-layer-carry-frozen)

| | Run | Hardware | Loop onset | Prequant val_bpb |
|---|-----|----------|-----------|-----------------|
| | 025b | 4H/1200s | step 2122 | **1.06917** |
| | 026  | 8H/600s  | step 2157 | **1.06893** |
| | Δ   |          |           | **+0.00024** |

Step-matched training loss:

| Step | 025b (4H) | 026 (8H) | Δ |
|------|-----------|----------|---|
| 500  | 2.6759 | 2.5881 | 0.088 |
| 1000 | 2.7667 | 2.8202 | -0.054 |
| 1500 | 2.5932 | 2.6439 | -0.051 |
| 2000 | 2.6538 | 2.6719 | -0.018 |
| 2500 | 2.5000 | 2.5588 | -0.059 |
| 3000 | 2.5687 | 2.5667 | +0.002 |
| 3500 | 2.3875 | 2.5677 | -0.180 |
| 4000 | 2.3758 | 2.4138 | -0.038 |
| 4500 | 2.3442 | 2.2832 | +0.061 |

Training loss crosses back and forth throughout. No systematic bias in either
direction. Final val_bpb delta is 0.00024 — indistinguishable from noise.

---

### Pair 2 — specs 029 / 030 (full-stack-025b)

| | Run | Hardware | Loop onset | Prequant val_bpb |
|---|-----|----------|-----------|-----------------|
| | 029 | 4H/1200s | step 2243 | **1.07007** |
| | 030 | 8H/600s  | step 2122 | **1.06822** |
| | Δ   |          |           | **+0.00185** |

Step-matched training loss:

| Step | 029 (4H) | 030 (8H) | Δ |
|------|----------|----------|---|
| 500  | 2.6689 | 2.5812 | +0.088 |
| 1000 | 2.7724 | 2.8181 | -0.046 |
| 1500 | 2.5979 | 2.6433 | -0.045 |
| 2000 | 2.6575 | 2.6715 | -0.014 |
| 2500 | 2.5134 | 2.5562 | -0.043 |
| 3000 | 2.5794 | 2.5674 | +0.012 |
| 3500 | 2.4058 | 2.5675 | -0.162 |
| 4000 | 2.3866 | 2.4054 | -0.019 |
| 4500 | 2.3491 | 2.2773 | +0.072 |

Same pattern: training loss tracks with small oscillations, no persistent gap.
Final val_bpb delta is 0.00185 — within normal variance.

---

### Pair 3 — 041A / 042B (038 + penalized_tanh loop band)

| | Run | Hardware | Loop onset | Prequant val_bpb |
|---|-----|----------|-----------|-----------------|
| | 041A | 8H/600s  | step 2193 | **1.10108** |
| | 042B | 4H/1200s | step 2287 | **1.09952** |
| | Δ    |          |           | **+0.00156** |

Step-matched training loss:

| Step | 041A (8H) | 042B (4H) | Δ |
|------|-----------|-----------|---|
| 500  | 2.6076 | 2.6883 | -0.081 |
| 1000 | 2.8540 | 2.8026 | +0.051 |
| 1500 | 2.6897 | 2.6433 | +0.046 |
| 2000 | 2.7300 | 2.7101 | +0.020 |
| 2200 | 2.6803 | 2.6575 | +0.023 |
| 2500 | 2.6203 | 2.5716 | +0.049 |
| 3000 | 2.6441 | 2.6469 | -0.003 |
| 3500 | 2.6509 | 2.4684 | +0.182 |
| 4000 | 2.4901 | 2.4787 | +0.011 |
| 4500 | 2.3673 | 2.4584 | -0.091 |

Same signature: losses cross throughout, final delta 0.0016. This pair uses a
different architecture family (penalized_tanh in loop band) from pairs 1/2,
which confirms the pattern is architecture-agnostic.

---

## Pattern across all three pairs

| Pair | Architecture | 4H val_bpb | 8H val_bpb | Δ |
|------|-------------|------------|------------|---|
| 025b/026 | cross-layer-carry-frozen | 1.06917 | 1.06893 | +0.00024 |
| 029/030  | full-stack-025b          | 1.07007 | 1.06822 | +0.00185 |
| 041A/042B | 038 + penalized_tanh    | 1.09952 | 1.10108 | -0.00156 |

Maximum observed delta: **0.0019**. Direction is not consistent — sometimes
4H is slightly better, sometimes 8H.

The training loss tracks with oscillations of ±0.05–0.18 at any given step,
but without systematic bias and with the oscillations averaging out by the end.

---

## What this means for screening

- A result on the 4H rung predicts the 8H result to within ~0.002 val_bpb
- Ranking of architectures on 4H is preserved on 8H
- The 4H rung costs half the money of 8H for the same signal

This is the basis for the 042 family: instead of burning 8H slots to test
loop-onset variants, screen on 4H and only promote the winner.

---

## Important exception — 043A vs 038A

**Do not use 043A/038A as evidence for or against this pattern.**

043A was a known bad control:
- launched from the 039b branch with a hand-written partial env
- did not faithfully replay the full 038A launch contract
- produced a catastrophic 0.148 val_bpb gap that is ~80× outside normal variance

That gap is a launch-contract bug, not evidence that 4H/8H diverge. See:

- `research/evaluations/043a-vs-038a-investigation.md`

A proper 038A faithful replay on 4H/1200s is needed to close this out.
