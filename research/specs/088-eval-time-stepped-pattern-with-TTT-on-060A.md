# Spec 088 — Eval-time matched-compute stepped pattern (073-shape) WITH TTT on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W9)
**Branch:** `exp/071-loop-pattern` (same as 080-087)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster AA1/AA2 W6
extended with TTT for leaderboard relevance.

## Hypothesis

Sibling to 087 (matched-compute stepped pattern eval, TTT off). 087
tested whether the trained model handles a `2-3-3-2 across {3,4,5,6}`
shape at matched compute. 088 enables TTT and GPTQ — the leaderboard-
relevant test.

Pattern body: `1,2,3,4,5,3,4,5,4,5,6,6,7,8` (same as 087).
17 total layer-passes (matched canonical compute). Layer visits:
- 3: 2× (vs canonical 3×)
- 4: 3× (matches canonical)
- 5: 3× (matches canonical)
- 6: 2× (vs canonical 1×)
- All others: 1× each

Rounds out the matched-compute pattern × TTT grid:

| Spec | Compute | Shape | TTT |
|---|---|---|---|
| 060A | 17 | canonical 3,3,3,1 | on |
| 085 | 17 | broad 2,2,3,2,2 | off |
| 086 | 17 | broad 2,2,3,2,2 | on |
| 087 | 17 | stepped 2,3,3,2 | off |
| **088** | **17** | **stepped 2,3,3,2** | **on** |

Reading 086 + 088 together (both TTT-on at matched compute, different
shapes): which non-canonical shape does TTT prefer?

## Baseline

060A canonical post-TTT post-quant val_bpb (same as 083/084/086).

## Expected Δ vs 060A canonical post-TTT

Tighter prediction range than 086 since stepped shape is closer to
canonical (only one layer's visits shifted vs 086's broader 5-layer
spread):

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (stepped + TTT compose) | -0.0005 to -0.0015 | low-medium |
| Plausible (TTT compensates for slight shape OOD) | -0.0003 to +0.0005 | medium |
| Likely (stepped 2-3-3-2 close enough to canonical that TTT mostly absorbs) | -0.0003 to +0.0008 | medium |
| Worst (TTT distorted by layer 6 doubled visit) | +0.0008 to +0.0025 | low-medium |

Higher likelihood of "neutral" outcome than 086 since stepped shape
is gentler.

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Win-conditional-on-087:** if 087 wins, 088 should win at least
  as much (or compose additively).
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0010

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,4,5,6,6,7,8"   # same as 087
NUM_LOOPS    = 2
ENABLE_LOOPING_AT = 0.0
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 1
PHASED_TTT_ENABLED = 3
PHASED_TTT_NUM_PHASES = 3
PHASED_TTT_PREFIX_DOCS = 2500
TTT_LORA_RANK = 80
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

Index lists (verified earlier in 073/087):
- `encoder_indices = [0, 1, 2, 3, 4, 5, 3, 4]` (8 indices)
- `decoder_indices = [5, 4, 5, 6, 6, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-087.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. Two distinct compiled functions
(forward_logits, forward_ttt) each compile once on first respective
invocation with the new index lists. Both produce one graph variant
each (different from canonical, different from 086's broad-pattern
graph variants — these are stepped-pattern variants). From the second
invocation of each onward, no recompile. **No mid-run recompile.**

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged
from 060A defaults.

Bank weights load from 060A checkpoint via `load_state_dict`. Layer-
indexed reads. Layer 6's bank reused on its second visit in this
pattern; layer 8 gets one visit (matches canonical post-loop).
**No weight slicing in compile.** **No narrow-K matmul.** **No
dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min total. Same breakdown as 086 (matched compute = same
  per-step pace as canonical):
  - Compile burst 1 (forward_logits): ~3-4 min
  - Compile burst 2 (forward_ttt): ~3-4 min
  - Pre-quant eval at stepped pattern: ~5-6 min
  - GPTQ calibration + quantization: ~3-5 min
  - Post-quant eval + TTT 3 phases: ~5-7 min

Total: 25-30 min. Cost: ~$3-4.

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/088-eval-stepped-pattern-TTT/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log
- `final_model.int6.ptz` — submittable artifact at the stepped pattern
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.075 → kill
- Compile time on either compiled function > 6 min → kill
- NaN → kill
- Total wallclock > 35 min → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

**Recommended order:**
1. Run 087 first (TTT-off stepped) → ~10 min, ~$1
2. Run 088 (TTT-on stepped) → ~25-30 min, ~$3-4

If 087 catastrophically loses (kill range), abort 088 — TTT won't
fix a fundamentally broken pattern.

## Reading 086 + 088 together

Both TTT-on at matched compute. They test which non-canonical SHAPE
is more robust under TTT:

| Spec | Layers reset from canonical | Visits added | Shape philosophy |
|---|---|---|---|
| 086 | 3,4 visits reduced; 6,7 visits doubled | layer 6 +1, layer 7 +1 | "broad spread" |
| 088 | 3 visits reduced; 6 visits doubled | layer 6 +1 only | "minimal canonical extension" |

If 086 wins and 088 loses: broad spread is the signal; canonical-
proximity hurts.
If 088 wins and 086 loses: minimal-extension is the signal; broad-
spread hurts.
If both win or both lose at similar magnitude: shape effect is real
but direction-symmetric.

## Followup specs gated on result

- If 088 wins: 089 candidate = TTT-on with stepped pattern at NL=3
  equivalent (stretched stepped) to compound the gain.
- If 088 ≈ canonical: stepped shape is interchangeable post-TTT.
- If 088 loses: matched-compute stepped shape thread closes.

## See also

- `research/specs/087-eval-time-stepped-pattern-on-060A.md` — TTT-off sibling
- `research/specs/086-eval-time-broad-pattern-with-TTT-on-060A.md` — broad-shape TTT cousin
- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — TTT-on template
- `research/ideas/parallelize-deep-looks.md` cluster AA + W9
