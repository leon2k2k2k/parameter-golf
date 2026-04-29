# Spec 094 — Eval-time TTT_LORA_LR=5e-5 (smaller LR) at canonical NL on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W15)
**Branch:** `exp/071-loop-pattern` (same as 080-093)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster YY1 (W14).

## Hypothesis

Sibling to spec 091 (more TTT phases). Both probe the **TTT
hyperparameter space at canonical recurrence**. 091 tested phase
count; 094 tests LoRA learning rate.

`TTT_LORA_LR=0.0001` is the default in 060A. This is the LR that
the TTT optimizer (Adam) uses for LoRA weight updates during the
3 phases. The canonical value came from 060A's tuned config, but
may not be optimal.

**Hypothesis:** if the canonical TTT_LORA_LR is over-aggressive (too
fast adaptation, possibly overfitting to per-doc prefix), reducing
to 5e-5 should yield a slight improvement. If it's right-tuned, the
smaller LR will under-adapt and bpb will worsen.

This spec runs at TTT_LORA_LR=5e-5 (half the canonical 1e-4) to
measure the gradient direction of this lever.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (canonical LR was over-aggressive; 5e-5 helps) | -0.0005 to -0.0015 | medium-low |
| Plausible (LR doesn't matter much in 3 phases) | -0.0001 to +0.0005 | medium-high |
| Likely (canonical LR is right-tuned; smaller under-adapts) | +0.0005 to +0.0015 | medium |
| Worst (severe under-adaptation) | +0.0015 to +0.0030 | low |

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0015

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = ""           # canonical contiguous-band logic
NUM_LOOPS    = 2
LOOP_START   = 3
LOOP_END     = 5
ENABLE_LOOPING_AT = 0.0
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 1
PHASED_TTT_ENABLED = 3
PHASED_TTT_NUM_PHASES = 3
PHASED_TTT_PREFIX_DOCS = 2500
TTT_LORA_RANK = 80
TTT_LORA_LR = 5e-5         # ← THE LEVER (canonical 1e-4)
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

(All other env vars per 060A defaults.)

## Code changes

**None.** `TTT_LORA_LR` already exists in 060A baseline config; just
set to a non-default value.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. `TTT_LORA_LR` is a Python
float passed to the TTT Adam optimizer. **Adam's optimizer state and
hyperparameters live entirely on the Python side**, NOT in the
compiled graph. Changing the LR has zero effect on the compiled
forward_logits or forward_ttt graphs.

**Zero new graph variants from this lever change.** **No mid-run
recompile possible.**

The compiled forward graphs are identical to 060A canonical (no
LOOP_PATTERN, contiguous-band logic). Per `loop_warmup`, both graphs
are pre-compiled during model setup.

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min. Same as 060A canonical (no extra compute):
  - Compile bursts: ~6-8 min (same as canonical)
  - Pre-quant eval: ~5-6 min
  - GPTQ: ~3-5 min
  - Post-quant + 3 TTT phases: ~5-7 min

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/094-eval-smaller-TTT-LR/seed_42/`.

Key artifacts:
- `train.log`
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (sanity)
- Post-TTT val_bpb > 1.075 → kill
- NaN → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

This spec is independent of 080-093. Tests a different lever
(TTT-LR) at canonical recurrence. Can run in any order.

For a complete TTT-LR sweep, complement with:
- Spec 095 candidate: `TTT_LORA_LR=2e-4` (larger LR, YY2 from W14).
- Together with canonical (1e-4) gives a 3-point LR scan.

## Reading 091 + 094 alongside canonical

| Lever varied | Spec | Direction |
|---|---|---|
| Phase count | 091 | up (3 → 4) |
| LoRA learning rate | **094** | **down (1e-4 → 5e-5)** |

Each samples a different TTT hyperparameter direction. Reading
together: which TTT hyperparameter was suboptimal in 060A's tuning?
- If 091 wins: phase count was the lever; 060A under-adapted.
- If 094 wins: LR was over-aggressive; 060A overshot.
- If both win: both directions were suboptimal; tuned config wins.
- If neither wins: 060A's canonical TTT config is approximately
  Pareto-optimal in this hyperparameter space.

## Followup specs gated on result

- If 094 wins: spec 095 = TTT_LORA_LR=2.5e-5 (push further).
- If 094 ≈ canonical: TTT-LR axis closes; pivot to other axes.
- If 094 loses: confirm canonical LR is right-tuned; explore
  larger-LR (095 with 2e-4).

## See also

- `research/specs/091-eval-time-more-TTT-phases-on-060A.md` — sibling
  TTT-axis spec (phase count)
- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — TTT-on
  template
- `research/ideas/parallelize-deep-looks.md` cluster YY (W14)
