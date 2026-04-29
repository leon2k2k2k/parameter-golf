# Spec 111 — Anderson acceleration on the recurrence loop

**Status:** FROZEN — code committed at `e9da01a` on `exp/111-anderson-recurrence`,
pushed to fork. Ready to run.

**Date:** 2026-04-30 (architectural research thread)
**Branch:** `exp/111-anderson-recurrence`
**Pinned commit:** `e9da01aa3dd64919ac07eb67ffee94ebabb394b9`
**Parent:** 060A (#1855 port; forks from `exp/060-resume-ckpt @ a0a48b7`).

## Hypothesis

Genuine architectural change to **how the recurrence iteration combines
its history**. Where the canonical loop runs a Markov-chain iteration
(each pass uses only the previous pass's output):

```
x_0 = (input from pre-loop)
x_1 = f(x_0)         # pass 0
x_2 = f(x_1)         # pass 1
x_3 = f(x_2)         # pass 2 → output
```

This spec replaces the iteration rule with **Anderson acceleration**
(Anderson 1965; Bai et al. 2019 for DEQ context):

```
At each iteration k, maintain a buffer of the last m+1 (x_i, f(x_i)) pairs.
Compute residuals g_i = f(x_i) - x_i.
Solve a tiny least-squares for mixing weights α minimizing ‖Σ α_i · g_i‖².
Take the next iterate as a weighted blend of recent f-outputs and x-inputs.
```

In words: instead of always walking in the latest direction the loop
function points, look at the last m+1 directions and walk in the
weighted combination that minimizes residual.

**Mathematical claim:** if `f` is a smooth contraction, Anderson
converges super-linearly (error squares per step), vs vanilla's
linear (error scales by a constant per step). Empirically: **3
Anderson passes ≈ 8-10 vanilla passes** on smooth fixed points.

Memory + literature support:
- Spec 064b's measured loop tax + 2509.23314's "step sizes shrink
  rapidly" finding → our recurrence is approximately a contraction.
- DEQ literature has used Anderson for years on similar iterative
  refinement problems.

## What this is testing

**Can we get NL=8-equivalent recurrence quality at NL=2 wallclock cost?**

The model still does **3 forward passes through layers 3, 4, 5** per
sequence (same FLOPs in the loop band). What changes is how the
intermediate outputs combine. If our `f` is contractive (which the
training-time-decay finding suggests), Anderson should compress 8-10
vanilla iterations of value into our 3 budgeted passes.

If the math holds in our setup: ~0.005 bpb improvement is plausible
(roughly the gap between NL=2 and NL=4 if iteration value scales
linearly in passes).

## Baseline

060A canonical post-TTT post-quant val_bpb (eval-only first pass).
Then 060A trained from scratch with Anderson if eval-only shows
signal.

## Expected Δ vs 060A

Two phases:

### Phase 1 — Eval-only on 060A trained checkpoint

The trained 060A weights expect Markov-chain iteration. Replacing the
iteration rule with Anderson at eval is OOD for the trained model —
the model never saw blended outputs at eval. So:

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (Anderson reaches a "better" fixed point even with trained-Markov weights) | -0.0010 to -0.0030 | low-medium |
| Plausible (slight regression from OOD iteration rule) | +0.0001 to +0.0010 | medium |
| Likely (trained model rejects the modified iteration) | +0.0010 to +0.0030 | medium-high |

If eval-only signal is plausible/best: phase 2 (training from scratch).

### Phase 2 — Train from scratch with Anderson iteration

The model trains with Anderson iteration; weights tuned for the
mixed-output regime.

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (Anderson buys NL=8-equivalent quality) | -0.0030 to -0.0080 | low |
| Plausible (modest convergence speedup) | -0.0010 to -0.0030 | medium |
| Likely (gain present but smaller) | -0.0005 to -0.0010 | medium |
| Worst (training instability from least-squares) | +0.0010 to +0.0050 | low |

## Accept criteria

### Phase 1 (eval-only on 060A)
- **Promote to phase 2:** post-TTT val_bpb ≤ canonical + 0.0010 (i.e., Anderson at eval doesn't catastrophically break trained model)
- **Abort:** post-TTT > canonical + 0.0010

### Phase 2 (training)
- **Win:** post-TTT val_bpb ≤ canonical − 0.0010
- **Noise:** ±0.0005
- **Kill:** ≥ canonical + 0.0010

## Architectural design

### Iteration rule

Standard:
```
for k in 0..num_loops:
    x = f(x)        # update in place; only most recent x matters
```

Anderson (with m=2 history, suitable for our 3-pass budget):
```
# Initialize history buffers (zeros, will be filled in first 2 passes)
x_history = [None] * 3        # x_0, x_1, x_2
f_history = [None] * 3        # f_0, f_1, f_2

for k in 0..num_loops:
    f_k = f(x_k)
    x_history[k] = x_k
    f_history[k] = f_k
    
    if k == 0:
        x_{k+1} = f_k                              # vanilla for first iter
    else:
        # Compute residuals available so far
        residuals = [f_history[i] - x_history[i] for i in 0..k]
        # Solve least-squares for mixing weights α (sum to 1)
        # Minimize ‖Σ α_i · residual_i‖²
        α = solve_least_squares(residuals)
        # Anderson update: weighted mix of f-outputs (β=1)
        x_{k+1} = sum(α_i * f_history[i] for i in 0..k)
```

For our 3-pass loop (NUM_LOOPS=2 → 3 passes), Anderson uses:
- Pass 0: vanilla (no history)
- Pass 1: m=1 (one residual)
- Pass 2: m=2 (two residuals) → **the meaningful Anderson step**

### Least-squares solve

For m=2 history (3 residuals), the least-squares problem is:

```
α* = argmin ‖Σᵢ αᵢ · gᵢ‖²    s.t. Σ αᵢ = 1
```

Solved analytically: `α = (G^T G)^(-1) · 1 / (1^T (G^T G)^(-1) · 1)`,
where G is the matrix of stacked residuals.

For m=2, this is a 3×3 matrix inversion. **Tiny — ~10K float ops vs
the ~10⁸ FLOPs per layer-application.** Negligible compute.

Implementation: solve in fp32 inside the compiled forward, with
`torch.linalg.solve` and a regularization fallback for ill-conditioned
cases (when residuals are nearly collinear / iteration converged).

### State carried across passes

A buffer `residual_buffer` of shape `(m+1, B, T, d)` per loop layer.
For m=2 with B×T tokens × 512 channels × bf16: ~3 GiB across all loop
layers (we have ~40 GiB headroom).

Stored as `register_buffer` (not param), zero-init. Updated in-place
each pass via modular indexing on `pass_idx % (m+1)`.

## Config diff vs 060A canonical

```
ANDERSON_ENABLED = 1
ANDERSON_HISTORY = 2     (m, the buffer depth — uses last 2 residuals)
ANDERSON_BETA = 1.0      (mixing factor — 1 = pure Anderson; <1 = blend with vanilla)
ANDERSON_REGULARIZATION = 1e-8   (added to LS matrix diagonal for stability)

NUM_LOOPS = 2            (canonical, 3 passes)
LOOP_START = 3, LOOP_END = 5
ENABLE_LOOPING_AT = 0.35
```

## Code changes

Branch: `exp/111-anderson-recurrence` (forks from 060A code @ a0a48b7).

### Files modified

- `records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py`:
  - New env vars (above).
  - New module `AndersonAccelerator(model_dim, history)` — manages residual buffer + least-squares solve.
  - In `GPT.__init__`: if `ANDERSON_ENABLED`, instantiate one `AndersonAccelerator` per loop layer.
  - In the loop iteration in `_forward_hidden`: replace `x = block(x)` (sequential) with the Anderson-updated rule.
  - Same for `forward_ttt`.

### Estimated LOC

- env vars + flag: ~10
- `AndersonAccelerator` module + LS solve helper: ~50
- `_forward_hidden` integration: ~20
- `forward_ttt` mirror: ~15
- Total: ~95 LOC

### Compile-graph audit

The hard part. **Stateful buffer across passes inside compile region.**
Done correctly:

1. **Buffer is `register_buffer`**, fixed shape `(m+1, B, T, d)`. Zero-init once at __init__ or first forward.
2. **Pass index is a Python int** counter incremented per call. Trace-time constant per graph, so each pass produces its own specialized graph variant — but only ONE per (pass_idx mod m+1), and there are only m+1 such states. Bounded variant set, all warmed at compile.
3. **Modular indexing** for buffer write: `buffer[k % (m+1)] = current_residual`. Indices are static at trace time.
4. **Least-squares solve** uses `torch.linalg.solve` on a tiny matrix. Standard PyTorch op, compile-friendly.
5. **Regularization** added to the LS matrix diagonal as a fixed scalar — no graph-variance.
6. **β scalar** is a learned `nn.Parameter` (or fixed at 1.0 — env var). Either way: static dispatch, no recompile.

Compile burst on first eval: m+1 = 3 graph variants compile (one per pass position in the cyclic buffer). After warmup: cached. **No mid-run recompile possible.**

### Always-tensor compliance

- Buffer always exists (shape pre-allocated).
- LS solve always runs (regularization handles ill-conditioning).
- No None passthroughs.
- `pass_idx` is a Python int; the graph variant it triggers is bounded at module construction.

**This is the riskiest compile audit in the 080-111 series.** I've written
it to be safe per the rules but it would benefit from a careful smoke run
on 4×H100 before promoting to 8×H100.

## Hardware ladder

### Phase 1 — eval-only on 060A trained checkpoint

- **Single rung: 4×H100, eval-only via RESUME_FROM_CKPT.**
- Wall: ~12-15 min (full pipeline including TTT + GPTQ).
- Cost: ~$2.

### Phase 2 — fresh training (only if Phase 1 promotes)

- **Mini rung: 4×H100, 5-min smoke** — required (substantial code change).
- **Official rung: 8×H100, 1 seed (42), 600s train + ~400s eval.**
- Cost: ~$6.

## Seed plan

Phase 1: 1 seed (42, matches 060A).
Phase 2: 1 seed (42) for screening; multi-seed if win.

## Stop-early criteria

- LS solve produces NaN/Inf → kill, regularize harder.
- Compile fail or recompile mid-run → kill, debug audit.
- Pre-quant val_bpb > 1.080 at step 5000 → kill.

## Cost estimate

- Phase 1: ~$2
- Phase 2 (gated on Phase 1): mini ~$1 + official ~$5 = ~$6

## Followup specs

- If 111 phase 2 wins: try m=3 history, β<1 mixing factor sweep.
- If 111 phase 1 catastrophically fails: confirm Markov-trained weights cannot accommodate Anderson; close eval-only thread; consider phase 2 only if motivated by other findings.

## See also

- `research/specs/110-multi-stream-345-543.md` — sibling architectural spec
- `research/ideas/parallelize-deep-looks.md` cluster K (W0)
- DEQ literature: Bai et al. 2019 "Deep Equilibrium Models"; Anderson 1965 "Iterative procedures for nonlinear integral equations"
