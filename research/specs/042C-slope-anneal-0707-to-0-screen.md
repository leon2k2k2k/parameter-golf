# Spec 042C — slope anneal 0.707→0.0 (BLUEPRINT — NOT RUNNABLE)

**Slug:** `042C-slope-anneal-0707-to-0-screen`
**Created:** 2026-04-26
**Status:** **BLUEPRINT** — parked until continuous-slope code path lands. Do **NOT** launch as-is.
**Branch:** `exp/042-slope-anneal-screen` (when realized)
**Commit:** TBD (requires code change — see "Blocked on" below)
**Links to:** `research/ideas/slope-annealing-sqrt-to-half.md`

## Why this is parked

Like 042B, this spec describes a **continuous slope anneal**. The target is
0.7071 → 0.0 (pure squared ReLU). Continuous interpolation requires
`NEGATIVE_SLOPE` as a runtime Triton arg, which OOMs at H100 with
`num_stages=4`. Same blocker as 042B; same fix needed
(`num_stages=3` + revert constexpr).

If we ran 042C as-is on the current codebase (`aff2de4`), the slope path would
fire as a STEP switch from 0.7071 → 0.0 at frac=0.25, not the gradual decay
described in the hypothesis. That's a different experiment entirely (more
abrupt; no gradual sparsification). Don't run that by accident.

## Blocked on

Same as 042B: implement continuous slope path
(`num_stages=3` + runtime `NEGATIVE_SLOPE` + linear interpolation in training loop).
See 042B blueprint for details.

## Hypothesis (still applies once unblocked)

Same mechanism as 042A but anneals all the way to `NEGATIVE_SLOPE=0.0` (pure
squared ReLU) instead of 0.5. Two effects compound:

1. **Phase-1 benefit (same as 042A):** Running at s=0.707 for the first 75%
   gives more gradient flow through negative activations, especially around
   loop activation at frac=0.35.

2. **Phase-2 sparsity bonus:** Pure `relu_square` (s=0) produces ~80-95%
   natural activation sparsity in the FFN intermediate. NVIDIA H100 has native
   2:4 structured sparse tensor core support — the sparse activations could
   accelerate FFN GEMMs ~1.3-1.5×, squeezing more effective steps into the
   1200s wallclock cap.

Unlike 042A (step switch), the slope would decay continuously — linearly from
0.7071 to 0.0 over the first 50% of warmdown (frac 0.25→0.625), then hold at
0.0 for the remainder. Half of warmdown lets the model fully adapt to pure
relu_square before the final convergence push.

## Config diff (when realized)

```bash
NEGATIVE_SLOPE=0.7071
SLOPE_WARMDOWN=0.0
```

All loop/architecture settings identical to baseline (layers 3-5, NL=2, frac=0.35).

## Regime (when realized)

- `4×H100`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=0`
- Regions: AP-JP-1 or US-NE-1
- Expect ~5–10% slower step time vs 042A due to `num_stages=3`

## Launch form (parked — do not run)

Same env vars as 042A with `SLOPE_WARMDOWN=0.0` and
`RUN_ID="042C-slope-anneal-0707-to-0"`. Not reproduced here to prevent
accidental copy-paste before the code is ready.

## Acceptance (when realized)

Baseline (039b): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641**
- **Noise zone**: 1.0641–1.0670
- **Kill**: ≥ **1.0670**

Watch quant damage closely — pure relu_square may interact differently with
GPTQ + LQER than the leaky variant.

## Prediction (when realized)

Two paths to win:
- If sparsity speedup materializes → more steps in same wallclock → guaranteed
  improvement vs 042A by ~0.001–0.003 bpb.
- If sparsity doesn't materially help → tied or marginally worse than 042A
  due to lost early-train flexibility.

## When to consider unblocking

- If 042A wins decisively (< 1.0641) → 042C is the natural follow-up.
- If 042A is noise/kill → 042C unlikely to help; deprioritize.
