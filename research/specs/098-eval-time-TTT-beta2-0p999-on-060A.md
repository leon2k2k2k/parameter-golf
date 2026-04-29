# Spec 098 — Eval-time TTT_BETA2=0.999 (more variance smoothing) at canonical NL on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W19)
**Branch:** `exp/071-loop-pattern` (same as 080-097)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster MMM1 (W18,
reframe of BBB2 from W15).

## Hypothesis

Sibling to spec 096 (TTT_BETA1=0.9). 096 tested adding momentum to
the gradient term; 098 tests increasing variance smoothing on the
second moment.

`TTT_BETA2=0.99` in 060A canonical is unusual — Adam's standard
default is 0.999. 060A's lower β2 means the variance estimate
adapts faster to local gradient magnitude (less smoothing).

This spec sets `TTT_BETA2=0.999` (standard Adam default). Together
with 096 (β1: 0→0.9), this independently tests both Adam β
parameters' canonical-vs-standard settings:

| Spec | β1 | β2 | Canonical-vs-standard |
|---|---|---|---|
| 060A | 0.0 | 0.99 | both unusual |
| 096 | **0.9** | 0.99 | β1 → standard |
| **098** | **0.0** | **0.999** | **β2 → standard** |

Combined with 096, identifies which Adam β was the load-bearing
canonical choice (or both, or neither).

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (β2=0.99 was suboptimal) | -0.0005 to -0.0015 | low-medium |
| Plausible (β2 has minor effect) | -0.0001 to +0.0005 | medium-high |
| Likely (β2=0.99 was tuned for a reason) | +0.0005 to +0.0020 | medium |
| Worst (severe variance smoothing breaks adaptation) | +0.0020 to +0.0040 | low |

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0020

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = ""
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
TTT_LORA_LR = 1e-4
TTT_BETA1 = 0.0           # canonical (096 tested 0.9)
TTT_BETA2 = 0.999         # ← THE LEVER (canonical 0.99)
TTT_WEIGHT_DECAY = 0.5    # canonical
TTT_BATCH_SIZE = 64       # canonical
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

## Code changes

**None.** `TTT_BETA2` is an existing env-var-readable parameter.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. `TTT_BETA2` is a Python
float passed to the TTT Adam optimizer constructor. **Adam state and
hyperparameters live entirely on the Python side**, NOT in the
compiled graph.

When β2=0.999, Adam maintains a longer-running variance estimate;
when β2=0.99, the estimate adapts faster. Either way, Adam updates
happen in eager mode and **the compiled forward graphs are unchanged**.

**Zero new graph variants from this lever.** **No mid-run recompile
possible.**

The compiled forward_logits and forward_ttt graphs are identical to
060A canonical (no LOOP_PATTERN; contiguous-band logic). All graphs
pre-compiled during loop_warmup.

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min (same as 060A canonical).

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/098-eval-TTT-beta2-0p999/seed_42/`.

Key artifacts:
- `train.log`
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.075 → kill
- NaN → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

This spec is independent of 080-097. Tests TTT-Adam-β2 axis. Test
in any order.

For a complete Adam-β-mapping, complement with:
- 096: β1: 0.0 → 0.9 (FROZEN)
- 098: β2: 0.99 → 0.999 (this spec)
- Joint: spec 099 candidate = both at standard (β1=0.9, β2=0.999).

## Reading 091/094/095/096/097/098 together (TTT hyperparameter map)

| Lever | Spec | Direction tested |
|---|---|---|
| Phase count | 091 | up |
| LoRA LR | 094/095 | down/up |
| Momentum (β1) | 096 | up (0→0.9) |
| Variance smoothing (β2) | **098** | **up (0.99→0.999)** |
| Batch size | 097 | down |

If 098 wins: standard Adam β2 was missing; canonical β2=0.99 was
suboptimal.
If 098 ≈ canonical: TTT is robust to β2; either value works.
If 098 loses: β2=0.99 was tuned for fast variance adaptation in 3
phases; standard 0.999 too smooth.

Reading 096+098 jointly:
- Both win: standard Adam params better; canonical β1+β2=(0,0.99) was
  doubly wrong.
- Both lose: canonical (β1=0, β2=0.99) was tuned (joint Pareto).
- Asymmetric: only one β was load-bearing.

## Followup specs gated on result

- If 098 wins: spec 099 candidate = β1=0.9, β2=0.999 (both standard)
- If 098 ≈ canonical: β2-axis closes.
- If 098 loses: confirm canonical β2; close that direction.

## See also

- `research/specs/096-eval-time-TTT-beta1-0p9-on-060A.md` — sibling β1 axis
- `research/specs/091/094/095/097` — sibling TTT-axis specs
- `research/ideas/parallelize-deep-looks.md` cluster MMM (W18) /
  reframed in W19 decisions
