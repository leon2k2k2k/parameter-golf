# Spec 090 — Eval-time band-position shift to {4,5,6} (one layer later) on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint. No new code.

**Date:** 2026-04-29 (autonomous wake W11)
**Branch:** `exp/071-loop-pattern` (same as 080-089)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster GG2 (W8) /
PP1 (W11).

## Hypothesis

Sibling to 089 on the position axis. 089 shifted the loop band to
{2,3,4} (one layer earlier than canonical). 090 shifts it to {4,5,6}
(one layer later). Together these bracket the canonical {3,4,5} band
on both sides:

| Spec | Loop band | Direction |
|---|---|---|
| 089 | {2,3,4} | one layer earlier |
| 060A | {3,4,5} | canonical |
| **090** | **{4,5,6}** | **one layer later** |

Pattern body: `4,5,6,4,5,6,4,5,6` — 9 visits.
Combined with pre `[0,1,2,3]` + post `[7,8,9,10]`: **17 total layer-passes**.

Layer visits:

| Layer | 060A canonical | 090 shifted | Δ |
|---|---|---|---|
| 0,1,2 | 1 each | 1 each | 0 |
| **3** | **3** | **1** | **−2** (now pre-loop) |
| 4 | 3 | 3 | 0 |
| 5 | 3 | 3 | 0 |
| **6** | **1** | **3** | **+2** (now in loop band) |
| 7 | 1 | 1 | 0 |
| 8,9,10 | 1 each | 1 each | 0 |

Net: **swap layer 3 out of the loop band; layer 6 in.** Symmetric to
089's swap of layer 5 out / layer 2 in.

**Mechanistic theory:** if there's a "right side" of the canonical band
to extend recurrence into, 089 and 090 will lose by different
magnitudes. If both lose similarly, the canonical band is positionally
sweet-spot.

In transformer depth interpretation: layers 4-6 are typically deeper
features. Shifting recurrence later concentrates iteration on more
abstract representations. May or may not transfer well from layer-3
trained recurrence.

## Baseline

Same as 089: 060A pre-quant post-EMA val_bpb at canonical NL=2
(~1.06358).

## Expected Δ vs 060A

Similar prediction to 089:

| Outcome | Δ bpb | Likelihood |
|---|---|---|
| Best (later band integrates abstract features better) | -0.0005 to +0.0005 | low-medium |
| Likely (band-position-sensitive; shift hurts) | +0.0010 to +0.0030 | medium-high |
| Worst (layer 6 wasn't trained for recurrence; destabilizes) | +0.0030 to +0.0080 | medium |

Reading 089 + 090 together is the main value. Symmetry diagnostic.

## Accept criteria

- **Win:** pre-quant val_bpb ≤ **1.06330**
- **Noise:** 1.06330 – 1.06430
- **Kill:** ≥ 1.06430

## Config diff vs 060A baseline

```
LOOP_PATTERN = "4,5,6,4,5,6,4,5,6"
NUM_LOOPS    = 2
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.0
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
```

Index lists (verified):
- `encoder_indices = [0, 1, 2, 3, 4, 5, 6, 4]` (8 indices)
- `decoder_indices = [5, 6, 4, 5, 6, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-089.

### Compile-graph audit

Eval-only run. Model construction reads `LOOP_PATTERN` once at
`GPT.__init__`, building 17-element index lists baked-in. The
iteration order shifts the loop band one position later. Compiled
`forward_logits` traces once on first eval batch — one new graph
variant. From the second eval batch onward, graph is cached. **No
mid-run recompile.**

Bank weights load from 060A checkpoint via `load_state_dict`. Layer 6's
bank — trained for single-visit feedforward — is now read three times
per forward. Each read uses the same loaded weights. **No weight
slicing in compile.** **No narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.** No TTT, no GPTQ.
- Wall: ~10 min.

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/090-eval-band-shift-456/seed_42/`.

Key artifacts:
- `train.log`
- `final.json` — pre-quant val_bpb at the shifted band

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Compile time > 6 min → kill
- NaN → kill

## Cost estimate

~$1 (eval-only, ~10 min on 4×H100).

## Reading 089 + 090 + 060A as a positional triplet

This is the primary value of 090 — completing the position-axis
diagnostic:

| Spec | Band | Hypothesis test |
|---|---|---|
| 089 | {2,3,4} | extend recurrence to shallower features |
| 060A | {3,4,5} | canonical (reference) |
| **090** | **{4,5,6}** | **extend recurrence to deeper features** |

Symmetry outcomes:
- **Symmetric loss:** 089 ≈ 090 in magnitude → canonical band is
  positionally optimal at this exact location.
- **Asymmetric loss (089 worse than 090):** later-band-shift is more
  forgiving than earlier-shift. Maybe deeper layers are more
  recurrence-amenable. Could motivate fresh-training spec at canonical
  + later extension.
- **Asymmetric loss (090 worse than 089):** earlier-band-shift is
  more forgiving. Shallow features tolerate iteration better. Less
  expected.
- **Both win/near-noise:** trained recurrence transferable across
  band positions — opens 2D position × shape sweep.

## Followup specs gated on result

- If 089 + 090 both lose ~0.001-0.003: canonical band positionally
  tuned. Closes positional thread; pivot to other axes.
- If asymmetric: spec 091 candidate = explore the favored direction
  further (e.g., {5,6,7} if 090 was the better side).
- If both win: write a position×shape combined sweep.

## See also

- `research/specs/089-eval-time-band-shift-234-on-060A.md` — direct sibling
- `research/specs/085/086/087/088` — matched-compute shape variants
- `research/timelines/loop-recurrence-2026-04-27.md` — historical
  position findings (#1726 et al.)
- `research/ideas/parallelize-deep-looks.md` cluster PP (W11)
