# Spec 093 — Eval-time band-shift {4,5,6} WITH TTT on 060A (leaderboard cell)

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W14)
**Branch:** `exp/071-loop-pattern` (same as 080-092)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster VV2 (W13/W14).

## Hypothesis

Sibling to 090 (band-shift {4,5,6} eval, TTT off) and to 092 (TTT-on
band-shift {2,3,4}). Together with 092, completes the position × TTT
grid for the leaderboard-relevant test.

Pattern body: `4,5,6,4,5,6,4,5,6` (same as 090). 17 total layer-passes.
Loop band shifted from canonical {3,4,5} to {4,5,6}.

Position × TTT grid:

| Band | TTT off | TTT on |
|---|---|---|
| {3,4,5} canonical | 060A pre-quant | 060A canonical (reference) |
| {2,3,4} earlier | 089 | 092 |
| **{4,5,6} later** | **090** | **093** |

Reading 092+093 jointly: **does TTT exhibit the same position
sensitivity as the underlying recurrence?** If 092 ≈ 093 in their
deltas vs canonical, TTT compensates symmetrically. If asymmetric,
there's a "favored direction" for band shifts under TTT.

## Baseline

060A canonical post-TTT (the leaderboard number for seed 42).

## Expected Δ vs 060A canonical post-TTT

Same range as 092 (mirror situation):

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (TTT recovers most of 090's loss) | -0.0005 to +0.0005 | low |
| Plausible (TTT partial recovery) | +0.0005 to +0.0020 | medium |
| Likely (TTT cannot fix OOD recurrence) | +0.0020 to +0.0050 | medium-high |
| Worst (TTT distorted) | +0.0050 to +0.0100 | medium |

If asymmetric vs 092: tells us which direction TTT prefers.

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0020

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "4,5,6,4,5,6,4,5,6"   # same as 090
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

Index lists (verified earlier in 090):
- `encoder_indices = [0, 1, 2, 3, 4, 5, 6, 4]` (8 indices)
- `decoder_indices = [5, 6, 4, 5, 6, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-092.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. Two distinct compiled
functions (forward_logits, forward_ttt) each compile once on first
respective invocation with the new band-shifted index lists. Both
produce one new graph variant each (different from canonical and
from 092's earlier-shift graphs). From the second invocation of
each onward, no recompile. **No mid-run recompile.**

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Bank weights load from 060A checkpoint. Layer 6's bank — visited
3 times in this pattern — reuses the loaded layer-6 weights without
modification on each visit. **No weight slicing in compile.** **No
narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min total. Same breakdown as 092:
  - Compile bursts: ~6-8 min combined
  - Pre-quant eval: ~5-6 min
  - GPTQ: ~3-5 min
  - Post-quant eval + 3 TTT phases: ~5-7 min

Cost: ~$3-4.

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/093-eval-band-shift-456-TTT/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log
- `final_model.int6.ptz` — submittable artifact at the shifted band
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.080 → kill
- Compile time on either compiled function > 6 min → kill
- NaN → kill
- Total wallclock > 35 min → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

**Recommended order:**
1. Run 090 (TTT-off band {4,5,6}) first — ~$1, ~10 min.
2. Run 093 (TTT-on band {4,5,6}) only if 090 was within plausible
   bounds.

## Reading 092 + 093 + 089 + 090 (the position × TTT 2x2)

Question: **does TTT compensate symmetrically for band-shift
direction?**

| Band | TTT off | TTT on | Δ(TTT-off) | Δ(TTT-on) |
|---|---|---|---|---|
| {3,4,5} | 060A (ref) | 060A (ref) | 0 | 0 |
| {2,3,4} | 089 | 092 | (089's Δ) | (092's Δ) |
| {4,5,6} | 090 | **093** | (090's Δ) | (**093's Δ**) |

Outcomes:
- **TTT symmetric:** 092 vs 089 ≈ 093 vs 090. TTT recovers (or fails
  to recover) similarly in both directions.
- **TTT direction-asymmetric:** 092 < 093 vs canonical or vice versa.
  The "favored direction" for band-shift is informative; future
  fresh-training specs could exploit it.
- **TTT band-amplifying:** both 092 and 093 worse than 089 and 090
  respectively. TTT actively distorts at OOD bands. Closes the
  position-axis as a viable lever.

## Followup specs gated on result

- If 092+093 both fail: position lever is fundamental; no TTT recovery.
  Pivot to other axes.
- If asymmetric: explore the favored direction further (e.g., {1,2,3}
  if {2,3,4} was favored).
- If both win: 094 candidate = explore wider symmetric shifts.

## See also

- `research/specs/090-eval-time-band-shift-456-on-060A.md` — TTT-off sibling
- `research/specs/092-eval-time-band-shift-234-with-TTT-on-060A.md` — paired position
- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — TTT-on template
- `research/ideas/parallelize-deep-looks.md` cluster VV (W13)
