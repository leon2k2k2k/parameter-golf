# Spec 102 — Eval-time compound broad pattern + 4 TTT phases on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W23)
**Branch:** `exp/071-loop-pattern` (same as 080-101)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster ZZZ1 (W23).

## Hypothesis

Third compound spec. Where 100 and 101 tested **compute × TTT**
compounds, this spec tests **shape × TTT**:

- 085 (broad pattern at matched 17-pass compute) +
- 091 (PHASED_TTT_NUM_PHASES=4)

Pattern body: `1,2,3,3,4,5,4,5,6,5,6,7,7` (same as 085).
17 total layer-passes. Layers 3..7 visited 2-3× each (broader band
than canonical {3,4,5}).

**Mechanistic theory:** the broad pattern shifts recurrence into
layers 6,7 that the trained model only saw once each. Extra TTT
phases give the LoRA more time to adapt to this OOD recurrence
pattern. If TTT can amortize the shape OOD-ness, the compound wins.

If 085 lost without TTT extension and 091 won at canonical shape:
102 reads as "TTT-axis amplification of a slightly-OOD shape." If
102 ≈ 086 (broad pattern + canonical TTT), the extra phase doesn't
add value beyond what canonical TTT already provides for this shape.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

Conditional on 085 and 091 individually:

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (broad pattern + extra TTT compose) | -0.0010 to -0.0025 | low |
| Plausible (one helps, sub-additive) | -0.0005 to -0.0010 | medium |
| Neutral | -0.0002 to +0.0005 | medium |
| Likely (broad shape OOD; extra TTT can't fix) | +0.0005 to +0.0020 | medium-high |

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0010**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0020

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "1,2,3,3,4,5,4,5,6,5,6,7,7"   # broad pattern (same as 085)
NUM_LOOPS    = 2
LOOP_START   = 3
LOOP_END     = 5
ENABLE_LOOPING_AT = 0.0
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 1
PHASED_TTT_ENABLED = 3
PHASED_TTT_NUM_PHASES = 4    # ← LEVER 2 (canonical 3, same as 091)
PHASED_TTT_PREFIX_DOCS = 2500
TTT_LORA_RANK = 80
TTT_LORA_LR = 1e-4
TTT_BETA1 = 0.0
TTT_BETA2 = 0.99
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

Index lists (verified earlier in 074/085 spec design):
- `encoder_indices = [0, 1, 2, 3, 3, 4, 5, 4]` (8 indices)
- `decoder_indices = [5, 6, 5, 6, 7, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` and `PHASED_TTT_NUM_PHASES` env vars
as 080-100.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. Two distinct compiled
functions (forward_logits, forward_ttt) each compile once on first
respective invocation with the broad-pattern index lists. Same
compile state as 086 (broad pattern + canonical TTT). **No mid-run
recompile.**

`PHASED_TTT_NUM_PHASES=4` is a Python loop bound; one extra TTT
phase call on the same compiled forward_ttt graph. **Zero new graph
variants from this lever.**

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Bank weights load from 060A checkpoint. Layers 6,7 visited twice
in this pattern — bank rows reused without modification. **No
weight slicing.** **No narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~30-32 min total. Same as 086 (matched compute) plus
  ~1-2 min for the extra TTT phase:
  - Compile bursts: ~6-8 min combined
  - Pre-quant eval: ~5-6 min
  - GPTQ: ~3-5 min
  - Post-quant eval + 4 TTT phases: ~7-9 min

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/102-eval-broad-pattern-4phases/seed_42/`.

Key artifacts:
- `train.log`
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.075 → kill
- Compile time on either compiled function > 8 min → kill
- NaN → kill

## Cost estimate

~$3-4 (full pipeline eval, ~30-32 min on 4×H100).

## Sequencing

**Recommended:** run 085 (broad pattern, TTT off) + 086 (broad
pattern + canonical TTT) + 091 (canonical NL + 4 phases) first.
102 worth running if at least 085 or 086 is within plausible band.

## Reading 100 + 101 + 102 + 086 (compound matrix)

| Spec | Variation | TTT extension | Compound axis |
|---|---|---|---|
| 100 | NL=3 eq | +phases | compute × TTT-phases |
| 101 | NL=3 eq | smaller LR | compute × TTT-LR |
| **102** | **broad shape** | **+phases** | **shape × TTT-phases** |
| 086 | broad shape | canonical | shape only (TTT canonical) |

102 vs 086: does adding the 4th phase amortize the broad-shape
OOD-ness?

## Followup specs gated on result

- If 102 wins: shape-axis composes with TTT extension. Spec 103
  candidate = stepped (087) + 4 phases.
- If 102 ≈ 086: extra phase doesn't help broad shape; pivot to
  other compounds.
- If 102 loses: TTT extension at OOD shape destabilizes; close
  shape-compound thread.

## See also

- `research/specs/085/086` — broad pattern single-axis specs
- `research/specs/091` — single-axis 4-phase TTT
- `research/specs/100/101` — sibling compound specs (compute × TTT)
- `research/ideas/parallelize-deep-looks.md` cluster ZZZ (W23)
