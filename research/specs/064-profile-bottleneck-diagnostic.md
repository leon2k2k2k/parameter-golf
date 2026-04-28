# Spec 064 — Bottleneck profiling diagnostic on 060A (no-fork)

**Status:** FROZEN — diagnostic run, no model changes, no fork, no
submission artifact. Runs the 060A pinned commit verbatim under an
external profiler wrapper. Output is a kernel-level trace + analysis
report.

**Harness committed:** `tmp_exec/launch_064_profile.sh` +
`tmp_exec/analyze_064_profile.py` (both syntax-checked locally).
Ready for execution to launch.

**Date:** 2026-04-29
**Branch:** **NONE** — 060A code is used as-is.
**Pinned commit:** `a0a48b7` (head of `exp/060-resume-ckpt`,
the 060A baseline).
**Parent run:** 060A (#1855 port, fresh-init training run).

## Hypothesis

The training run is compute-bound (100% GPU util, VRAM headroom).
The 33% throughput drop at loop activation (memory:
`project_throughput_step_function`) implies loop-block passes are
disproportionately expensive per layer-equivalent. We don't know
*which kernels* eat that budget. Knowing turns "cheaper loop
iterations" from a guess into a targeted intervention.

Three plausible bottlenecks, in roughly decreasing order of prior:

- **(A) Attention compute** dominates. Recurrent attention does more
  O(T²) work; this would surface as `flash_attn_func` topping the
  kernel ranking and growing post-loop.
- **(B) Kernel launch overhead** grows when the loop expands the
  layer-index list. More `block.forward` calls = more launches.
  Surfaces as a long tail of small kernels with launch-gap idle.
- **(C) Memory-bandwidth bound** on RMSNorm + residual writes; loop
  passes do these more often per token.

Whichever it is, the followup spec (065/066/067) is decided by the
ranking.

## Baseline

060A unchanged at commit `a0a48b7`. Single seed (42). Full model
architecture. **Do NOT swap to a small proxy** — the loop step
function only manifests at real scale, and that's the phenomenon
we're diagnosing.

## Expected output

Not a bpb number. Outputs:

1. **`profile_full.nsys-rep`** — Nsight Systems trace covering the
   full ~250 step run, captured externally via `nsys profile`.
   Includes kernel timeline, CUDA API calls, NVTX ranges,
   GPU memory transactions.
2. **`profile_full.sqlite`** — exported queryable form of the
   trace (via `nsys export`).
3. **`kernel_summary_pre_loop.txt`** — top kernels by wall time for
   the pre-loop window (extracted from sqlite via post-processing).
4. **`kernel_summary_post_loop.txt`** — same for post-loop window.
5. **`vram_curve.csv`** — `step, allocated_MiB, max_allocated_MiB`
   logged from the existing train.log (060A already prints this; we
   just scrape it).
6. **`toks_per_s.csv`** — same, from existing train.log.
7. **`analysis.md`** — human-written interpretation: hot kernels,
   loop-activation step function visible in the trace, which of
   {A,B,C} above is confirmed. Written by research after
   the run.

All saved to `runs/064-profile-bottleneck/seed_42/`.

## Accept criteria

- `profile_full.nsys-rep` exists, opens cleanly in Nsight Systems
  GUI (or on the CLI via `nsys stats`).
- Loop activation event clearly visible in the timeline (sharp
  step function in step-time around `enable_looping_at`).
- Top-3 kernels by wall time identified for both pre-loop and
  post-loop regimes.
- One concrete hypothesis-confirmed-or-rejected line in analysis.md.

## Config diff vs 060A

```
MAX_TRAIN_STEPS = 250          # short — just enough to span loop activation
WALLCLOCK_BUDGET = 240         # 240s, hard kill (200s + nsys overhead)
EMA_DECAY = 0.0                # disable EMA eval (not relevant; saves time)
PHASED_TTT_ENABLED = 0         # disable TTT (we're profiling training, not eval)
GPTQ_ENABLED = 0               # disable post-quant (no submission needed)
RUN_LABEL = profile_seed_42
```

(All other env vars verbatim from 060A.)

## Code changes

**None.** 060A's `train_gpt.py` is run verbatim. Profiling is
attached externally via the launch script.

`tmp_exec/launch_064_profile.sh` (new, ~40 lines):

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT=/workspace/runs/064-profile-bottleneck/seed_42
mkdir -p "$OUT"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline

# Inline prewarm + cache restore per project conventions
# (kept short — full prewarm not needed for a 250-step diagnostic)
bash /workspace/parameter-golf/tmp_exec/cache_restore.sh || true

# Capture full timeline including kernel-level CUDA activity.
# --duration 0 = capture full duration; we'll slice windows in post.
# --sample=cpu --backtrace=lbr for fine-grained CPU + GPU correlation.
nsys profile \
    --output="$OUT/profile_full" \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    torchrun --standalone --nproc_per_node=4 train_gpt.py \
    2>&1 | tee "$OUT/train.log"

# Export sqlite + extract kernel rankings windowed pre/post loop activation
nsys export --type=sqlite --output="$OUT/profile_full.sqlite" "$OUT/profile_full.nsys-rep"

python /workspace/parameter-golf/tmp_exec/analyze_064_profile.py \
    --sqlite "$OUT/profile_full.sqlite" \
    --train-log "$OUT/train.log" \
    --out-dir "$OUT"
```

Plus `tmp_exec/analyze_064_profile.py` (new, ~80 lines): post-run
analysis — queries the sqlite for kernel rankings in two time
windows (pre/post loop activation, derived from `train.log` step
times), writes `kernel_summary_*.txt`, scrapes VRAM and tok/s from
`train.log`, writes the CSVs.

Both new files land on `research` (no exp branch needed):
- `tmp_exec/launch_064_profile.sh`
- `tmp_exec/analyze_064_profile.py`

## Hardware ladder

- **Single rung: 4×H100, 1 seed (42), 250 steps max, 240s wallclock cap.**
- No mini rung (architecture unchanged; smoke unnecessary).

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths — train data, tokenizer. No hotstart; fresh init
is fine since we're not measuring model quality.

## Stop-early criteria

- `nsys profile` startup fails → kill, install nsys / fix env, retry
- tok/s drops to <50% of expected at any step (something is broken,
  not just slow)
- Step >300 (we've gone past the budget without loop activating —
  ENABLE_LOOPING_AT may be set wrong)
- Wallclock exceeds 240s (hard cap)

## Cost estimate

~$1.50 (4×H100 × 240s ≈ 17 min wallclock incl. pod startup).

## Open questions for interview

1. **Is `nsys` installed on the parameter-golf pod template
   (`y5cejece4j`)?** If absent, the launch script must
   `apt install nvidia-nsight-systems-cli` early. Add a check.
2. **`nsys` overhead.** Full-trace nsys typically adds 5-15%
   wallclock; with `--gpu-metrics-device=all` it can be more.
   This affects the *absolute* tok/s numbers but NOT the relative
   kernel ranking, which is what we care about. Note this in
   analysis.md.
3. **Pod region.** Standard NE-1 → JP fallback.
4. **What halts** if `profile_full.nsys-rep` is corrupted or the
   sqlite export fails? Halt and ask. The run is cheap to rerun;
   do not promote a corrupt trace.
5. **Stop pod after?** Yes. One-shot diagnostic.

## Followup specs gated on this analysis

- **Top kernel = `flash_attn_func` (or attn-related), grows
  post-loop:** spec 065 = cheaper loop attention (skip-attn loops,
  smaller per-pass head count, reduced KV).
- **Top kernel = MLP matmul, ratio steady pre/post:** we're at
  matmul roofline; pivot back to algorithmic levers.
- **Long tail of tiny kernels with launch gaps:** spec 066 =
  block-level kernel fusion or graph-mode forward.
- **RMSNorm + elementwise dominate:** spec 067 = fewer norms
  per pass / norm fusion.

## Why this is worth the $1.50

The 33% loop tax is the largest throughput finding in the project,
and we've never instrumented *which kernels* eat it. Every followup
spec gated on its result avoids wasting cycles on ideas that don't
attack the actual bottleneck. Self-paying by even one avoided null
spec.
