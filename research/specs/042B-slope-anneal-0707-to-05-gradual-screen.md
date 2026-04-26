# Spec 042B — slope anneal 0.707→0.5 gradual (BLUEPRINT — NOT RUNNABLE)

**Slug:** `042B-slope-anneal-0707-to-05-gradual-screen`
**Created:** 2026-04-26
**Status:** **BLUEPRINT** — parked until continuous-slope code path lands. Do **NOT** launch as-is.
**Branch:** `exp/042-slope-anneal-screen` (when realized)
**Commit:** TBD (requires code change — see "Blocked on" below)
**Links to:** `research/ideas/slope-annealing-sqrt-to-half.md`

## Why this is parked

This spec describes a **continuous (linear) slope anneal** from 0.7071 to 0.5
across the first half of warmdown. That requires `NEGATIVE_SLOPE` to be a
**runtime** Triton arg (not `tl.constexpr`), so the kernel can take a different
slope value each step without recompiling.

Our previous attempt at runtime float (commit `eb90ab1`) hit an **H100 shared
memory OOM**: with `num_stages=4` the matmul kernel needs 245,784 bytes of
shared memory but H100 caps at 232,448. We reverted to constexpr in `2593982`.

To realize this spec, we need a code change first:

## Blocked on

**Implement continuous slope path:**
1. Drop `num_stages` from 4 → 3 in `linear_leaky_relu_square_kernel` (frees
   ~49 KB of shared memory; ~5–10% throughput cost)
2. Restore `NEGATIVE_SLOPE` as a runtime arg (revert the `tl.constexpr` part
   of commit `2593982`)
3. Wire continuous interpolation in the training loop:
   ```python
   if h.slope_warmdown >= 0 and frac >= 1.0 - h.warmdown_frac:
       progress = (frac - (1.0 - h.warmdown_frac)) / (h.warmdown_frac * 0.5)
       progress = min(progress, 1.0)
       slope = (1 - progress) * h.negative_slope + progress * h.slope_warmdown
       for module in base_model.modules():
           if isinstance(module, MLP):
               module.negative_slope = slope
   ```

When that code lands, this spec gets a real commit hash, status flips to READY,
and the launch form below becomes valid.

## Hypothesis (still applies once unblocked)

Same target as 042A (0.707→0.5) but with a smooth linear decay instead of a step
switch. The slope decays from 0.7071 to 0.5 over the first 50% of warmdown
(frac 0.25→0.625), then holds flat at 0.5 for the rest. Gives the model time to
adapt gradually rather than experiencing an abrupt change.

Compare against 042A (step switch) to isolate whether gradual vs instant matters.

Schedule:
- frac 0.00–0.25: slope = 0.7071
- frac 0.25–0.625: slope linearly 0.7071→0.5
- frac 0.625–1.0: slope = 0.5 (flat)

## Config diff (when realized)

```bash
NEGATIVE_SLOPE=0.7071
SLOPE_WARMDOWN=0.5
# (when continuous path is in code, no extra env var needed —
# the same SLOPE_WARMDOWN value is used as the linear endpoint
# instead of a step target)
```

All loop/architecture settings identical to baseline (layers 3-5, NL=2, frac=0.35).

## Regime (when realized)

- `4×H100`, `SEED=42`, `MAX_WALLCLOCK_SECONDS=1200`, `TTT_ENABLED=0`, `TRAINING_ONLY_SCREEN=0`
- Regions: AP-JP-1 or US-NE-1
- Expect ~5–10% slower step time vs 042A due to `num_stages=3`

## Launch form (parked — do not run)

Same env vars as 042A with `RUN_ID="042B-slope-anneal-0707-to-05-gradual"`.
See 042A spec for the full env block. Once unblocked, copy 042A's launch form
verbatim and update RUN_ID + run-dir name.

## Acceptance (when realized)

Baseline (039b): pre-quant EMA val_bpb **1.06514**

- **Win**: pre-quant EMA val_bpb < **1.0641**
- **Noise zone**: 1.0641–1.0670
- **Kill**: ≥ **1.0670**

## Prediction (when realized)

Marginal improvement over 042A if gradual transition helps. More likely similar
result — the benefit of 0.707 pre-training dominates; transition smoothness is
secondary. Conservative: same as 042A (noise zone). Optimistic: ~1.063x.

## When to consider unblocking

- If 042A wins decisively → 042B is worth the code work to test whether
  smoother transition gains another ~0.001 bpb.
- If 042A is noise/kill → 042B unlikely to help; deprioritize.
- If H100 throughput becomes the binding constraint, the `num_stages=3` cost
  may not be justified for a marginal experiment.
