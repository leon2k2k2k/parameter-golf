# Spec 095 — Eval-time TTT_LORA_LR=2e-4 (larger LR) at canonical NL on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W16)
**Branch:** `exp/071-loop-pattern` (same as 080-094)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster YY2 (W14).

## Hypothesis

Sibling to spec 094 (TTT_LORA_LR=5e-5, smaller-LR direction). 095
tests the **larger-LR direction**: TTT_LORA_LR=2e-4 (2× canonical).
Together with 094 and the canonical 060A run, gives a **3-point LR
scan**: 5e-5 / 1e-4 / 2e-4.

`TTT_LORA_LR=0.0001` is the default in 060A. Larger LR (2e-4) means:
- Faster LoRA adaptation per phase
- More aggressive movement in the loss landscape
- Risk of overshoot or instability

If 094's smaller LR helped, the canonical was over-aggressive — and
095 with larger LR should hurt more. If 094 hurt, canonical was
right-tuned and 095 should hurt similarly. If 094 was neutral, 095
could go either way.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (canonical was under-aggressive; 2e-4 helps) | -0.0005 to -0.0015 | low |
| Plausible (LR insensitive in 3 phases) | -0.0001 to +0.0005 | medium |
| Likely (canonical right-tuned; larger overshoots) | +0.0005 to +0.0030 | medium-high |
| Worst (severe overshoot, instability) | +0.0030 to +0.0080 | medium |

Higher chance of significant regression than 094 — larger-LR sweeps
historically more prone to instability than smaller-LR.

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0030 (high regression
  threshold given larger-LR risks)

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
TTT_LORA_LR = 2e-4         # ← THE LEVER (canonical 1e-4, sibling 094: 5e-5)
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

## Code changes

**None.** Same env-var-only override as 094.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. `TTT_LORA_LR` is a Python
float passed to the TTT Adam optimizer. **Adam state and
hyperparameters live entirely on the Python side**, NOT in the
compiled graph.

**Zero new graph variants from this lever.** **No mid-run recompile
possible.**

The compiled forward graphs are identical to 060A canonical (no
LOOP_PATTERN; contiguous-band logic). All graphs pre-compiled during
loop_warmup.

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min (same as 060A canonical).

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/095-eval-larger-TTT-LR/seed_42/`.

Key artifacts:
- `train.log`
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.080 → kill (instability from larger LR)
- NaN → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

This spec is independent of 080-094. Test in any order; ideal
sequence: run 094 first, see direction, then decide 095.

## Reading 094 + 095 + canonical (3-point LR scan)

| Spec | TTT_LORA_LR | Direction |
|---|---|---|
| 094 | 5e-5 | half canonical |
| 060A | 1e-4 | canonical |
| **095** | **2e-4** | **double canonical** |

Curve shape:
- **Monotone increasing:** smaller is better; canonical too aggressive.
  Try 2.5e-5 (spec 096 candidate).
- **Monotone decreasing:** larger is better; canonical too conservative.
  Try 4e-4 (spec 096 candidate).
- **U-shaped with minimum at canonical:** 060A's tuning is right.
  Close LR axis.
- **Inverted U (canonical worst):** unusual; may indicate noise floor.

## Followup specs gated on result

- If 095 wins: extend to 4e-4.
- If 095 loses but 094 won: optimal LR is below canonical; refine
  downward.
- If both 094 and 095 lose: canonical LR is optimal.
- If both win: canonical was at a noisy point; LR axis is forgiving.

## See also

- `research/specs/094-eval-time-smaller-TTT-LR-on-060A.md` — direct sibling
- `research/specs/091-eval-time-more-TTT-phases-on-060A.md` — TTT-axis
  sibling on different parameter (phase count)
- `research/ideas/parallelize-deep-looks.md` cluster YY (W14)
