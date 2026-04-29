# Spec 085 — Eval-time matched-compute broad pattern (074-shape) on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint. No new code beyond 080-084 reuse.

**Date:** 2026-04-29 (autonomous wake W6)
**Branch:** `exp/071-loop-pattern` (same as 080-084)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: `parallelize-deep-looks.md`
cluster AA1 (W6).

## Hypothesis

Cousin to the 080/081/082 scaling-curve specs (which vary *compute*).
This spec varies **pattern shape** at **fixed compute** (17 passes =
canonical 060A). Tests how robust the trained iterative-refinement
function is to shape changes at runtime.

Pattern body: `1,2,3,3,4,5,4,5,6,5,6,7,7` (same as the spec 074
training-time variant, never run yet). Visit profile:

| Layer | 060A canonical | 085 broad pattern | Δ |
|---|---|---|---|
| 0,1,2 | 1 each | 1 each | 0 |
| 3 | 3 | 2 | −1 |
| 4 | 3 | 2 | −1 |
| 5 | 3 | 3 | 0 |
| **6** | **1** | **2** | **+1** |
| **7** | **1** | **2** | **+1** |
| 8,9,10 | 1 each | 1 each | 0 |

**Total layer-passes: 17 each.** Net: 060A's recurrence band {3,4,5}
is broadened to {3..7}, with 2 visits at layers 6 and 7 (which the
trained model only saw once each).

**Hypothesis:** the trained recurrence weights at layers 3,4,5 are
designed for "iterate this band 3× each." Asking layers 6,7 to take
the recurrent role is OOD. Two outcomes:

- **Robust:** weights at layers 6,7 are general enough that 2 visits
  refine instead of break. bpb ≈ canonical.
- **Brittle:** layers 6,7 weren't trained for recurrence; running them
  twice within the U-Net loop introduces spurious refinement noise.
  bpb worsens.

This is **dual to 080/081/082's compute axis**: instead of "more
passes through canonical {3,4,5}," it's "same total passes but
spread to a different set of layers."

## Baseline

Same as 080/081/082: 060A pre-quant post-EMA val_bpb at canonical
NL=2 (~1.06358).

## Expected Δ vs 060A

Tighter prediction range than usual since weights are seed-matched:

| Outcome | Δ bpb | Likelihood |
|---|---|---|
| Best (recurrence robust to band shape) | -0.0005 to +0.0005 | medium |
| Likely (slight worsening from OOD layer 6,7 iteration) | +0.0005 to +0.0020 | medium-high |
| Worst (brittle, layers 6,7 destabilize) | +0.0020 to +0.0050 | low-medium |

Less likely to win than the deeper-recurrence specs (080/081) — but
still informative either way.

## Accept criteria

- **Win:** pre-quant val_bpb ≤ **1.06330** (Δ ≤ −0.00028)
- **Noise:** 1.06330 – 1.06430
- **Kill:** ≥ 1.06430

## Config diff vs 060A baseline

```
LOOP_PATTERN = "1,2,3,3,4,5,4,5,6,5,6,7,7"
NUM_LOOPS    = 2          # any positive value enables looping_active
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.0   # eval-only
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
```

Matches the spec 074 training-time pattern exactly, but at eval-time on
the 060A checkpoint.

Index lists (verified earlier in the 074 spec design):
- `encoder_indices = [0, 1, 2, 3, 3, 4, 5, 4]` (8 indices)
- `decoder_indices = [5, 6, 5, 6, 7, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-084.

### Compile-graph audit

Eval-only run. Model construction reads `LOOP_PATTERN` once at
`GPT.__init__`, building the 17-element index lists baked-in at
module construction. Compiled `forward_logits` traces once on the
first eval batch — one new graph variant due to the different
iteration *order* (encoder/decoder lists differ from canonical even
though total length matches). From the second eval batch onward,
graph is cached. **No mid-run recompile.**

The bank weights load from 060A checkpoint; bank indexing by layer
identifies which weights to use at each visit. Layer 6 and 7
weights are loaded as for the trained model and applied during
their second visit in the body without modification. **No weight
slicing.** **No narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.** No TTT, no GPTQ.
- Wall: ~10 min (compile burst + eval at canonical-compute pace).

## Seed plan

1 seed (42).

## Inputs / Outputs

- Same as 080. Output dir: `/workspace/runs/085-eval-broad-pattern/seed_42/`.

Key artifacts:
- `train.log` — eval log
- `final.json` — pre-quant val_bpb at the broad pattern

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (broad pattern broke model)
- Compile time > 6 min → kill
- NaN → kill

## Cost estimate

~$1 (eval-only, ~10 min on 4×H100).

## Reading 080-085 together

| Spec | Compute axis | Shape axis | TTT |
|---|---|---|---|
| 060A | NL=2 (17 passes) | canonical | reference (with TTT for leaderboard) |
| 080 | NL=3 (20 passes) | canonical-extended | off |
| 081 | NL=4 (23 passes) | canonical-extended | off |
| 082 | NL=1 (11 passes) | no recurrence | off |
| 083 | NL=3 (20 passes) | canonical-extended | on |
| 084 | NL=4 (23 passes) | canonical-extended | on |
| **085** | **NL=2 (17 passes)** | **broad {3..7}** | **off** |

085 fills the "fixed-compute, different-shape" cell. Reading 085
alongside 080/082 disentangles "more compute" from "different shape" —
helps decide whether the trained model rewards depth or shape changes.

## Followup specs gated on result

- If 085 wins or near-noise: matched-compute pattern shape is a real
  lever. Spec 086 candidate = TTT-on version of 085.
- If 085 plausibly worse: trained recurrence is band-specific
  ({3,4,5}). Closes broader-band hypothesis at eval.
- Either way: provides a calibration point for AA3 (multi-shape
  sweep, if it happens).

## See also

- `research/specs/074-loop-pattern-broad-17pass.md` — training-time
  variant of the same pattern (tests "trained from scratch with this
  pattern")
- `research/specs/080/081/082/083/084` — sibling eval-only specs
- `research/ideas/parallelize-deep-looks.md` cluster AA (W6)
