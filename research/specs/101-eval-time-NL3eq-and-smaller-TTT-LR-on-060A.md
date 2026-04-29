# Spec 101 — Eval-time compound NL=3 eq + TTT_LORA_LR=5e-5 on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W22)
**Branch:** `exp/071-loop-pattern` (same as 080-100)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster WWW1 (W22).

## Hypothesis

Second compound spec. Combines 080's deeper recurrence (NL=3 eq via
LOOP_PATTERN) with 094's smaller TTT LR (5e-5).

**Mechanistic theory:** deeper recurrence (more layer-passes per
sequence) amplifies the chain-rule gradient magnitude during TTT
backward. A smaller LR may compensate — stabilizing TTT adaptation
under the larger gradients of deeper iteration.

This is **a different compound axis from spec 100** (NL=3 eq + 4
TTT phases). Together, 100 and 101 explore which TTT lever
composes better with deeper recurrence:
- 100: more phases (more adaptation cycles)
- 101: smaller LR (more stable adaptation per cycle)

If 101 wins more than 100: stability matters more than cycles.
If 100 wins more than 101: cycles matter more than stability.
If both win: TTT axis is forgiving and any extension helps.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

Conditional on 080 and 094 individually winning:

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (constructive composition) | -0.0010 to -0.0025 | low |
| Plausible (sub-additive) | -0.0005 to -0.0010 | medium |
| Neutral (one helps) | -0.0002 to +0.0005 | medium |
| Destructive | +0.0005 to +0.0020 | low-medium |

If 094 lost, 101 unlikely to win.

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0010**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0010

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,3,4,5,3,4,5,6,7"   # NL=3 eq (same as 080)
NUM_LOOPS    = 2
LOOP_START   = 3
LOOP_END     = 5
ENABLE_LOOPING_AT = 0.0
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 1
PHASED_TTT_ENABLED = 3
PHASED_TTT_NUM_PHASES = 3        # canonical
PHASED_TTT_PREFIX_DOCS = 2500
TTT_LORA_RANK = 80
TTT_LORA_LR = 5e-5               # ← LEVER 2 (canonical 1e-4, same as 094)
TTT_BETA1 = 0.0                  # canonical
TTT_BETA2 = 0.99                 # canonical
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

## Code changes

**None.** Same `LOOP_PATTERN` and `TTT_LORA_LR` env vars as 080-100.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. Two distinct compiled
functions (forward_logits, forward_ttt) each compile once on first
respective invocation with the NL=3-eq index lists. Same compile
state as 083 / 100. **No mid-run recompile.**

`TTT_LORA_LR=5e-5` is a Python float passed to Adam optimizer. Adam
state lives Python-side, NOT in compiled graph. **Zero new graph
variants from the LR lever.**

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Bank weights load from 060A checkpoint. Layers 3,4,5 visited 4
times each — bank rows reused without modification. **No weight
slicing.** **No narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~30-32 min total. Same as 083/100 (compile + NL=3 eq +
  3 TTT phases).

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/101-eval-NL3eq-smaller-LR/seed_42/`.

Key artifacts:
- `train.log`
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.075 → kill
- Compile time on either compiled function > 8 min → kill
- NaN → kill
- Total wallclock > 35 min → kill

## Cost estimate

~$3-4 (full pipeline eval, ~30-32 min on 4×H100).

## Sequencing

**Recommended:** run 080 (NL=3 eq, TTT-off) and 094 (smaller LR,
canonical NL) first. If both win, launch 101. If either loses
substantially, abort.

## Reading 100 + 101 + 080 + 094 together

Build a 2x2 NL × TTT-LR composition table:

| | TTT phases=3, LR=1e-4 (canonical) | TTT phases=4 (091/100) | TTT LR=5e-5 (094/101) |
|---|---|---|---|
| NL=2 | 060A | 091 | 094 |
| NL=3 eq | 083 | 100 | **101** |

Compound effects:
- 100 vs (083 + 091's gain): tests phase-count compound
- 101 vs (083 + 094's gain): tests LR-tuning compound
- 100 vs 101: which TTT lever composes better with deeper recurrence?

## Followup specs gated on result

- If 101 > 100 (smaller LR composes better than more phases):
  spec 102 = NL=3 eq + smaller LR + 4 phases (triple compound).
- If 101 < 100 by clear margin: phase-count is the right TTT-axis
  for deeper recurrence; pivot to TTT2 (NL=4 + 4 phases).
- If both lose: TTT axis doesn't compose with deeper recurrence;
  return to single-axis exploration.

## See also

- `research/specs/080-eval-time-deeper-loop-on-060A.md` — single-lever sibling
- `research/specs/094-eval-time-smaller-TTT-LR-on-060A.md` — single-lever sibling
- `research/specs/100-eval-time-NL3eq-and-4-TTT-phases-on-060A.md` — sibling compound (different TTT lever)
- `research/ideas/parallelize-deep-looks.md` cluster WWW (W22)
