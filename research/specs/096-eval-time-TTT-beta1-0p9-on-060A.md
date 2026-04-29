# Spec 096 — Eval-time TTT_BETA1=0.9 (add momentum to TTT Adam) at canonical NL on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W17)
**Branch:** `exp/071-loop-pattern` (same as 080-095)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster HHH1 (W17,
reframe of BBB1 from W15).

## Hypothesis

060A canonical sets `TTT_BETA1=0.0`, meaning the TTT Adam optimizer
applies *no* momentum to gradients during LoRA updates. Each LoRA
update is purely (gradient × LR), no smoothing across prior steps.

This is unusual — standard Adam practice uses β1=0.9 to dampen
gradient noise via exponential moving averaging. The canonical
β1=0 may be:
- (a) **A tuned choice** for the eval-time TTT regime (3 phases × short
  per-phase sequences), where momentum-based smoothing actively hurt.
- (b) **A stripped-down default** that wasn't tuned because gains were
  marginal.

This spec sets `TTT_BETA1=0.9` (standard Adam default) and runs the
full pipeline to test which interpretation is correct.

If (a): 096 worsens vs canonical → momentum hurts in this regime;
canonical is right.
If (b): 096 ≈ canonical or improves → momentum is neutral or
beneficial; canonical default just wasn't tuned.

This is a **novel hyperparameter axis** for the TTT lever space —
sibling to 091 (phase count) and 094/095 (LoRA LR). Together these
samples map out the TTT hyperparameter space.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (momentum smooths noise; canonical was suboptimal) | -0.0005 to -0.0015 | medium-low |
| Plausible (momentum has minor effect in 3 phases) | -0.0001 to +0.0005 | medium-high |
| Likely (momentum was actively bad; canonical right) | +0.0005 to +0.0020 | medium |
| Worst (momentum severely overshoots) | +0.0020 to +0.0050 | low |

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0020

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
TTT_LORA_LR = 1e-4
TTT_BETA1 = 0.9            # ← THE LEVER (canonical 0.0)
TTT_BETA2 = 0.99           # canonical (unchanged)
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

## Code changes

**None.** `TTT_BETA1` is an existing env-var-readable parameter in
060A's hyperparam dump (per memory `project_baseline_1851_post_1872`).

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. `TTT_BETA1` is a Python float
passed to the TTT Adam optimizer constructor. **Adam state and
hyperparameters live entirely on the Python side**, NOT in the
compiled graph.

When β1=0.9, Adam maintains a momentum buffer per parameter; when
β1=0.0, Adam skips momentum (or maintains a zero buffer). Either way
the **compiled forward graphs are unchanged** — Adam updates happen
in eager mode.

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

- Output dir: `/workspace/runs/096-eval-TTT-beta1-0p9/seed_42/`.

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

This spec is independent of 080-095. Test in any order. Reading
together with 091 (phase count) and 094/095 (LoRA LR) maps the TTT
hyperparameter space.

## Reading 091 + 094/095 + 096 (TTT hyperparameter axes)

| Lever | Spec(s) | Direction(s) tested |
|---|---|---|
| Phase count | 091 | up (3→4) |
| LoRA LR | 094/095 | down (5e-5) and up (2e-4) |
| **Momentum (β1)** | **096** | **up (0.0→0.9)** |

If 096 wins: TTT-momentum is a real lever; the canonical TTT config
was missing a free improvement. Notable since β1=0 is unusual.
If 096 ≈ canonical: TTT is robust to momentum; β1=0 is fine.
If 096 loses: canonical β1=0 was tuned; momentum harms in this
regime.

## Followup specs gated on result

- If 096 wins: explore other Adam params (TTT_BETA2 at 0.95 or 0.999).
- If 096 ≈ canonical: TTT-momentum-axis closes.
- If 096 loses: confirm canonical β1=0; close that direction.
- Either way: completes the TTT hyperparameter mapping started by
  091/094/095.

## See also

- `research/specs/091-eval-time-more-TTT-phases-on-060A.md` — sibling
  TTT-axis spec (phase count)
- `research/specs/094-eval-time-smaller-TTT-LR-on-060A.md` — sibling
  TTT-axis spec (LR down)
- `research/specs/095-eval-time-larger-TTT-LR-on-060A.md` — sibling
  TTT-axis spec (LR up)
- `research/ideas/parallelize-deep-looks.md` cluster HHH (W17)
