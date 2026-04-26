# Evaluation — Spec 042A (slope anneal 0.7071→0.5 at warmdown)

**Run dir:** `runs/042A-slope-anneal-0707-to-05-screen/`
**Commit:** `aff2de4` on `exp/042-slope-anneal-screen`
**Baseline:** `runs/039-neg-slope-screen-on-1797-base/baseline/` (pre-quant EMA val_bpb 1.06514, quantized 1.07410, 5156 steps)
**Eval date:** 2026-04-26 (revised same-day to fix baseline mis-citation)

> **Eval revision notes (later same day):** the original eval cited
> `slope_0p5_seed_42/` as the baseline, which is actually a 3-min early
> screen run (771 steps, EMA 1.258 — not a meaningful baseline). The
> "comparison arm" 039bA (EMA 1.07397) used by the train_loss table is
> ~0.009 worse than canonical baseline, so 042A's 0.013–0.027 train_loss
> advantage vs 039bA does NOT translate to a real win vs baseline. The
> step-matched comparison vs *canonical* baseline shows 042A is slightly
> *behind* through most of training and ends ~tied. Updated tables below.

## Result — noise zone (pre-quant), GPTQ recovered after fix

| metric | 039b baseline | 042A | Δ |
|---|---|---|---|
| pre-quant EMA val_bpb | 1.06514 | **1.06661** | +0.00147 |
| quantized val_bpb (original, broken) | 1.07410 | ~~2.05028~~ | ~~+0.97618~~ |
| **quantized val_bpb (042D-recovered, fixed)** | **1.07410** | **1.07579** | **+0.00169** |
| quant cost (Δ pre→post) | +0.00896 | +0.00918 | +0.00022 (clean) |
| final step (wallclock-limited) | 5156 | 5098 | −58 |
| step 5000 raw val_bpb | 1.0763 | 1.0778 | +0.0015 |

The original quantized val_bpb (2.05) was a deserialize bug, not a quant
quality issue. After the fix in commit `72c7328` and verification via
spec 042D (EVAL_ONLY=1, ~$0.25), the same quantized blob produces
**1.07579** — quant cost matches baseline almost exactly (+0.00918 vs
baseline's +0.00896, delta of +0.00022). The quant pipeline is healthy.

**Submission viability:** the existing 042A artifacts are now a valid
submission candidate (no retrain needed). Final number to compare for a
submission would be the post-quant 1.07579 vs baseline's 1.07410 → still
+0.00169 worse than baseline, still in noise zone.

Pre-quant EMA lands in the **noise zone** (acceptance: win <1.0641, noise 1.0641–1.0670, kill ≥1.0670). Quantized result is **catastrophically broken** — from 1.074 to 2.050, a +0.976 regression. The submission file would be unusable.

## Noise/signal judgment (pre-quant) — REVISED

The +0.00147 pre-quant delta is inside the noise zone. The original
revision of this eval claimed a systematic train_loss advantage based on
comparison to 039bA. That comparison was misleading because 039bA itself
ran ~0.009 worse than canonical baseline. **The correct step-matched
comparison vs canonical baseline is below.**

| Step | baseline | 042A | Δ (042A − baseline) |
|---|---|---|---|
| 1000 | 2.7670 | 2.7724 | +0.005 |
| 1500 | 2.5928 | 2.6020 | +0.009 |
| 2000 | 2.6552 | 2.6589 | +0.004 |
| 2500 | 2.5104 | 2.5117 | +0.001 |
| 3000 | 2.5784 | 2.5826 | +0.004 |
| 3500 | 2.4059 | 2.4116 | +0.006 |
| 4000 | 2.4050 | 2.4038 | −0.001 |
| 4500 | 2.3854 | 2.3836 | −0.002 |
| 5000 | 2.4015 | 2.4004 | −0.001 |

042A spends most of training (steps 300–3900) very slightly *behind*
canonical baseline. It catches up in late warmdown (4000–4500) and ends
essentially tied. Final pre-quant EMA: +0.00147 worse. **The slope
hypothesis did not produce a measurable win.**

### Why the original "buggy" 042A looked so much better

The pre-fix 042A (commit `2593982`, broken with recompile pause) ran:
- slope switch fired at step **1623** (frac 0.300, recompile-inflated)
- loop activated at step **1625** (frac 0.377, recompile-inflated)
- ended at step 3034 with EMA 1.0886

In the broken run, the recompile pause inflated `elapsed_ms` mid-pre-warm,
pushing both event triggers (`frac >= 0.25` and `frac >= 0.35`) to fire
much earlier in absolute step number than they should have. Loop activation
at step 1625 vs the fixed run's step 2289 = **664 extra loop steps** at
any given absolute step, which explains the 0.10–0.13 train_loss "advantage"
the broken run showed at matched steps. The slope=0.7071 contributed ≈0.

This is a clean negative result for the slope-anneal hypothesis as
formulated. The early-train signal in the broken run was an artifact of
loop-activation-timing being shifted, not the slope mechanism.

### Surviving hypothesis: earlier loop activation (separate spec)

The buggy run did suggest one real lever: **earlier loop activation**.
039bE tried this (`ENABLE_LOOPING_AT=0.175`) but with `lr_schedule_mode=
first_half_default_then_floor`, which confounds the result (final EMA
1.06916, worse than baseline). A clean isolation — earlier loop, normal
LR schedule, slope=0.5 throughout — is worth a separate spec (043A).

## Training mechanics — all verified clean

- **Pre-warm**: `slope_anneal: precompiled warmdown kernel slope=0.5000 (forward+forward_logits, both looping states)` — logged at startup. cache_size_limit=32 fix (commit `aff2de4`) confirmed effective; no mid-training recompile events anywhere in log.
- **Slope switch**: `slope_anneal: 0.7071→0.5000 step:1635 frac:0.250` — fired exactly at frac=0.25, zero throughput pause. Step 1600 at 4.9m, step 1700 at 5.2m — identical 0.3m/100-step interval through the switch.
- **Loop activation**: `layer_loop:enabled step:2289 frac:0.350` — normal ~7% throughput step-down (4300K→3900K tok/s), no recompile, no disruption.
- **Val trajectory**: smooth descent: 1.2375 → 1.2002 → 1.1579 → 1.1201 → 1.0778 at steps 1000–5000.

## GPTQ catastrophic failure — root cause

**Pre-quant EMA val_bpb: 1.06661. Quantized val_bpb: 2.050.**

The model after training has `module.negative_slope = 0.5` (switched at step 1635 and held through step 5098). But the launch script exported `NEGATIVE_SLOPE=0.7071` to the environment and that value is still live when GPTQ runs.

`NEGATIVE_SLOPE` is a `tl.constexpr` parameter in the leaky_relu_square Triton kernel. When GPTQ runs calibration forward passes, if the kernel dispatch re-reads `NEGATIVE_SLOPE` from the environment (or re-compiles with the env var), it uses slope=0.7071 for calibration while the model weights were optimized for slope=0.5 after the switch. This produces severely wrong activation statistics, making the GPTQ Hessians garbage, destroying the quantized model entirely.

**Fix required before re-run or submission:**

Option A (simple): Override `NEGATIVE_SLOPE` to the final slope before the GPTQ block executes — either in the training script or by exporting `NEGATIVE_SLOPE=$SLOPE_WARMDOWN` after training ends.

Option B (robust): Have the training script explicitly set the env var to `module.negative_slope` before GPTQ calibration begins, ensuring the Triton kernel and model state are always in sync.

The submission file size (15,917,057 bytes) was within limit — the GPTQ bug is a calibration error, not a compression failure.

## Decision — REVISED: KILL slope hypothesis; pivot to early-loop spec

After the baseline-comparison correction, the slope-anneal hypothesis is
*not* showing the win it appeared to. Net assessment:

1. **Slope-anneal hypothesis: KILL.** 042A is +0.00147 worse than canonical
   baseline at pre-quant EMA. Step-matched train_loss is slightly behind
   for most of the run. The "advantage" in the broken run came from
   recompile-shifted loop-activation timing, not from slope.
2. **GPTQ slope-mismatch bug: FIXED in `72c7328`** (in-tree fix to
   `deserialize()`). Verification spec 042D (1×H100, ~$0.25, EVAL_ONLY=1)
   pending — would recover the existing artifacts as a (slightly losing)
   submission candidate.
3. **042B (gradual slope to 0.5) and 042C (gradual to 0.0): keep as
   BLUEPRINT only.** Given the negative result on the step-switch version,
   gradual variants are even less likely to win, and they require
   continuous-slope code work (num_stages=3) that isn't justified.
4. **Pivot research to "early loop activation" as a clean, isolated spec
   (043A).** This is the actual mechanism the broken run was hinting at.
   Single env var change vs baseline (`ENABLE_LOOPING_AT=0.20`), no LR
   confound, no slope confound.

The infrastructure work from this arc IS durable: cache_size_limit=32,
deserialize slope sync, EVAL_ONLY=1 mode, forward_logits pre-warm.
All keep, all benefit future specs.

## Val trajectory (042A, complete)

| Step | val_bpb |
|---|---|
| 1000 | 1.2375 |
| 2000 | 1.2002 |
| 3000 | 1.1579 |
| 4000 | 1.1201 |
| 5000 | 1.0778 |
| 5098 (final) | 1.0778 |
| **pre-quant EMA** | **1.06661** |
| quantized | ~~2.050~~ (broken — see above) |

## Cost

~$4.15 (4×H100, ~21 min including diagnostic eval). Smoke run (042A-recompile-smoke): $1.50. Total for this hypothesis cycle: ~$5.65.

## Cross-references

- Spec: `research/specs/042A-slope-anneal-0707-to-05-screen.md`
- Predecessor smoke: `runs/042A-recompile-smoke/notes.md` (Dynamo LRU cache eviction diagnosis)
- Baseline: `runs/039-neg-slope-screen-on-1797-base/slope_0p5_seed_42/`
- Comparison arm: `runs/039-neg-slope-screen-on-1797-base/039bA/` (slope=0.5, same hardware)
