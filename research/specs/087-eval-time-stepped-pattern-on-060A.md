# Spec 087 — Eval-time matched-compute stepped pattern (073-shape) on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint. No new code.

**Date:** 2026-04-29 (autonomous wake W8)
**Branch:** `exp/071-loop-pattern` (same as 080-086)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: `parallelize-deep-looks.md`
cluster AA (W6) — sibling to 085.

## Hypothesis

Sibling to 085 (matched-compute broad pattern eval, TTT off). 085 tested
the **broad-spread shape** (5 layers each visited 2-3 times). 087 tests
the **stepped 2-3-3-2 shape** across {3,4,5,6} (canonical-similar but
extends one layer deeper into 6).

Pattern body: `1,2,3,4,5,3,4,5,4,5,6,6,7,8` (same as 073 training-time
spec). 17 total layer-passes (matched canonical compute).

| Layer | 060A canonical | 087 stepped | Δ |
|---|---|---|---|
| 0,1,2 | 1 each | 1 each | 0 |
| 3 | 3 | 2 | −1 |
| 4 | 3 | 3 | 0 |
| 5 | 3 | 3 | 0 |
| **6** | **1** | **2** | **+1** |
| 7,8 | 1 each | 1 each | 0 |
| 9,10 | 1 each | 1 each | 0 |

Net change vs canonical: **trade one visit at layer 3 for one at layer 6**.
The recurrence band effectively shifts from {3,4,5} to {3,4,5,6} but
with one fewer visit at layer 3.

**Hypothesis:** if the trained recurrence "rolls forward" cleanly into
layer 6 (which the trained model only saw once), the test should
land near canonical. If layer 3's third visit was load-bearing, this
loses.

Reading 085 + 087 together at TTT-off:

| Spec | Shape | What it tests |
|---|---|---|
| 060A | canonical 3,3,3 on {3,4,5} | reference |
| 085 | broad 2,2,3,2,2 on {3..7} | "spread visits wider" |
| **087** | **stepped 2,3,3,2 on {3,4,5,6}** | **"shift band one deeper, keep peak"** |

Different mechanisms tested by each:
- 085 spreads visits to MANY non-trained recurrence layers (6 AND 7)
- 087 only extends ONE layer deeper (6), keeps everything else
  closer to canonical
- Expected: 087 < 085 if "minimal departure from canonical" matters
  more than "broad coverage."

## Baseline

Same as 080/081/082/085: 060A pre-quant post-EMA val_bpb at canonical
NL=2 (~1.06358).

## Expected Δ vs 060A

Tighter prediction range than 085 since shape is closer to canonical:

| Outcome | Δ bpb | Likelihood |
|---|---|---|
| Best (gentle band extension is fine) | -0.0003 to +0.0005 | medium |
| Likely (slight worsening from layer 6 OOD) | +0.0005 to +0.0015 | medium-high |
| Worst (layer 6 destabilizes recurrence) | +0.0015 to +0.0030 | low |

## Accept criteria

- **Win:** pre-quant val_bpb ≤ **1.06330** (Δ ≤ −0.00028)
- **Noise:** 1.06330 – 1.06430
- **Kill:** ≥ 1.06430

## Config diff vs 060A baseline

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,4,5,6,6,7,8"
NUM_LOOPS    = 2
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.0   # eval-only
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
```

Index lists (verified earlier in 073 design):
- `encoder_indices = [0, 1, 2, 3, 4, 5, 3, 4]` (8 indices)
- `decoder_indices = [5, 4, 5, 6, 6, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-086.

### Compile-graph audit

Eval-only run. Model construction reads `LOOP_PATTERN` once at
`GPT.__init__`, building 17-element index lists baked in at module
construction. Compiled `forward_logits` traces once on first eval
batch — one new graph variant due to the different iteration order
from canonical (and from 085's broad pattern). From the second eval
batch onward, graph is cached. **No mid-run recompile.**

Bank weights load from 060A checkpoint via `load_state_dict`; layer-
indexed reads. Layer 6's bank loaded as for the trained model and
reused on its second visit in this pattern. **No weight slicing.**
**No narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.** No TTT, no GPTQ.
- Wall: ~10 min.

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/087-eval-stepped-pattern/seed_42/`.

Key artifacts:
- `train.log` — eval log
- `final.json` — pre-quant val_bpb at the stepped pattern

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Compile time > 6 min → kill
- NaN → kill

## Cost estimate

~$1 (eval-only, ~10 min on 4×H100).

## Sequencing with 085

- Run 085 first (broad pattern). 087 informative regardless of
  085's outcome — they test different mechanisms.
- If both 085 and 087 win or near-noise: pattern shape is genuinely
  flexible; promote a multi-shape sweep.
- If only 087 wins: gentle band extension works; broad spread doesn't.
  Spec 088 candidate = TTT-on version of 087.
- If only 085 wins: broad coverage matters more than canonical-band-
  similarity.
- If both lose: matched-compute pattern shape is closed.

## Reading 080-087 together

Full 2D map filling out:

| Spec | Compute | Shape | TTT |
|---|---|---|---|
| 060A | 17 | canonical | on |
| 080 | 20 | extended | off |
| 081 | 23 | extended | off |
| 082 | 11 | shallow | off |
| 083 | 20 | extended | on |
| 084 | 23 | extended | on |
| 085 | 17 | broad {3..7} | off |
| 086 | 17 | broad {3..7} | on |
| **087** | **17** | **stepped 2,3,3,2** | **off** |

At matched compute (17), three shapes tested at TTT-off: canonical
(060A), broad (085), stepped (087). Triangulates whether shape
matters at all.

## Followup specs gated on result

- If 087 wins: spec 088 = TTT-on stepped pattern (completes leaderboard
  cell for stepped shape).
- If 087 ≈ 085 (both small loss or noise): pattern shape is flexible;
  any matched-compute pattern roughly works.
- If 087 < 085 by clear margin: gentle extensions beat broad spreads;
  iterate further on stepped variants.

## See also

- `research/specs/085-eval-time-broad-pattern-on-060A.md` — sibling broad-pattern test
- `research/specs/073-loop-pattern-stepped-17pass.md` — training-time variant
  of this pattern
- `research/specs/080-eval-time-deeper-loop-on-060A.md` — TTT-off compute-axis
  template
- `research/ideas/parallelize-deep-looks.md` cluster AA (W6) and W8
  decisions
