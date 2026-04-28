# Spec 064 — Bottleneck profiling diagnostic on 060A

**Status:** DIAGNOSTIC — no model changes, no submission artifact.
Output is a kernel-level trace + analysis report.

**Date:** 2026-04-29
**Branch:** `exp/064-profile-bottleneck` (forked from `exp/060-resume-ckpt @ a0a48b7`)
**Pinned commit:** TBD (after launch-script + profiler harness commit)
**Parent:** 060A (#1855 port).

## Hypothesis

The training run is compute-bound (100% GPU util, VRAM headroom). The
33% throughput drop at loop activation (memory:
`project_throughput_step_function`) implies loop-block passes are
disproportionately expensive per layer — they should add
`3/total_layers ≈ 33%` more layer-equivalents per step, but they cost
more than that. Either:

- (A) Attention is the dominant cost and loop layers' attention is
  not amortizing well (most likely, given recurrent attention does
  more O(T²) work per token);
- (B) Kernel launch overhead grows when the loop expands the layer
  list (more block.forward calls = more launches);
- (C) Memory-bandwidth bound on loop layers' specific kernels
  (e.g. RMSNorm + residual writes dominate).

Whichever it is, knowing it changes which architectural lever is
worth chasing next. Specifically: if (A), the "cheaper loop
iterations" idea (memory: `project_cheaper_loop_iterations_for_tomorrow`)
becomes high priority — skip-attn loops or smaller per-pass attention
would directly attack the bottleneck.

## Baseline

060A unchanged. Single seed (42). Full model architecture (do NOT
swap to small proxy here — the loop step function only manifests at
real scale, and that's the phenomenon we're diagnosing).

## Expected output

Not a bpb number. Outputs:

1. `torch.profiler` trace (Chrome-trace JSON) covering 10 steps:
   5 pre-loop-activation, 5 post-loop-activation.
2. Per-block forward time table (CSV): block index × pre/post-loop ×
   mean ms.
3. Kernel-by-time-percent ranking (top 20) for both regimes.
4. Step-level VRAM curve over the full 200-step warmup.
5. tok/s curve at 10-step granularity.
6. Optimizer step + grad norm CSV.
7. Analysis writeup in `runs/064-profile-bottleneck/analysis.md`.

## Accept criteria

- Trace JSONs successfully captured for both regimes (no profiler
  crash).
- Top-3 kernels by time identified for each regime.
- Loop activation event clearly visible in tok/s curve.
- One concrete hypothesis-confirmed-or-rejected line in analysis.

## Config diff vs 060A

```
PROFILE_ENABLED          = 1   # turns on torch.profiler harness
PROFILE_STEPS_BEFORE     = 5   # steps captured pre-loop
PROFILE_STEPS_AFTER      = 5   # steps captured post-loop
PROFILE_OUTPUT_DIR       = /workspace/runs/064-profile-bottleneck/seed_42/
LOG_PER_BLOCK_TIMING     = 1   # CUDA-event timing per block.forward
LOG_VRAM_EVERY_N_STEPS   = 10
LOG_TOKENS_PER_S_EVERY_N = 10
MAX_TRAIN_STEPS          = 250 # short — just enough to span loop activation
WALLCLOCK_BUDGET         = 200 # 200s, hard kill
```

All other env vars unchanged from 060A. **Skip TTT, skip GPTQ, skip
EMA evaluation** — these are not relevant to the diagnostic and add
artifact-build time.

## Code changes

Three additive sites in `train_gpt.py` (no logic changes to model):

1. **Profiler harness around the training loop.** Wraps the step
   loop with `torch.profiler.profile(...)` configured with
   `schedule=schedule(wait=W, warmup=Wm, active=A, repeat=2)` such
   that one capture window lands ~10 steps before
   `enable_looping_at` and the second lands ~10 steps after. Saves
   to `chrome_trace_pre_loop.json` and `chrome_trace_post_loop.json`.

2. **Per-block CUDA-event timing.** In `_forward_hidden`, wrap each
   `self.blocks[i](...)` call with `torch.cuda.Event` start/end
   pairs (only when `LOG_PER_BLOCK_TIMING=1` to keep prod path
   clean). Append `(step, block_idx, layer_idx_in_bank, elapsed_ms)`
   to a CSV.

3. **VRAM + tok/s tracking.** Existing tok/s logging exists; add
   `torch.cuda.memory_allocated()` and `max_memory_allocated()`
   readout every 10 steps to a separate CSV.

Branch: `exp/064-profile-bottleneck`. Commit: TBD.

## Hardware ladder

- **Single rung: 4×H100, 1 seed (42), 250 steps max, 200s wallclock cap.**
- No mini rung (the architecture is unchanged; smoke is unnecessary).

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths — train data, tokenizer. No hotstart; fresh init
is fine since we're not measuring model quality.

## Artifacts emitted

To `/workspace/runs/064-profile-bottleneck/seed_42/`:

- `chrome_trace_pre_loop.json` — torch.profiler trace, ~5 steps before loop activation
- `chrome_trace_post_loop.json` — torch.profiler trace, ~5 steps after loop activation
- `per_block_timing.csv` — CUDA-event timing per block per step
- `vram_curve.csv` — `step, allocated_MiB, max_allocated_MiB`
- `toks_per_s.csv` — `step, tok_per_s, wallclock_s`
- `loss_grad_norm.csv` — `step, train_loss, grad_norm`
- `train.log` — full stdout/stderr
- `final_step_summary.json` — `{step_when_loop_activated, mean_toks_per_s_pre, mean_toks_per_s_post, top_kernels_pre, top_kernels_post}` (auto-generated post-run by `tmp_exec/analyze_064.py`)

**Do NOT save model checkpoints, EMA, or GPTQ artifacts.** This is a
diagnostic run — no submission blob is needed and skipping them
shortens the run.

## Stop-early criteria

- Profiler crash → kill, debug locally
- tok/s drops to <50% of expected at any step (something is broken,
  not just slow)
- Step >300 (we've gone past the budget without loop activating —
  ENABLE_LOOPING_AT may be set wrong)
- Wallclock exceeds 200s (hard cap)

## Cost estimate

~$1.50 (4×H100 × 250s ≈ 17 min wallclock, including pod startup).

## Open questions for interview

1. **Pod region.** Standard NE-1 → JP fallback. Diagnostic is short
   enough that capacity issues are unlikely to bite.
2. **Profiler overhead.** torch.profiler with stack-traces adds
   ~10-20% wallclock per captured window. Captured windows are 10
   steps total out of 250 — so overall wallclock impact <5%. The
   tok/s number reported in our analysis should EXCLUDE the profiled
   windows (which are inflated by profiler overhead).
3. **What halts** if the trace files are corrupted or the profiler
   schedule misses the loop activation event? Halt and ask. The run
   is cheap to rerun; do not promote a corrupt trace to analysis.
4. **Stop pod after?** Yes. This is a one-shot diagnostic; no
   followup launches expected immediately.

## Followup specs gated on this analysis

- If attention dominates pre + post: **spec 065** = cheaper loop
  attention (skip-attn loops, smaller per-pass head count).
- If launch overhead dominates: **spec 066** = block-level kernel
  fusion or graph-mode forward.
- If RMSNorm + elementwise dominate: **spec 067** = fewer norms per
  pass / norm fusion.
- If nothing surprising: confirms compute-bound on matmul roofline,
  pivot back to algorithmic levers (depth recurrence is the right
  bet).

## Why this is worth the $1.50

The 33% loop tax is the single largest throughput finding in this
project. We've never instrumented *which kernels* eat that budget.
Knowing this turns "cheaper loop iterations" from a guess into a
targeted intervention. The diagnostic is also self-paying: every
followup spec gated on its result avoids wasting cycles on ideas
that don't attack the actual bottleneck.
