# Spec 089 — Eval-time band-position shift to {2,3,4} (one layer earlier) on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint. No new code.

**Date:** 2026-04-29 (autonomous wake W10)
**Branch:** `exp/071-loop-pattern` (same as 080-088)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster GG1 (W8).

## Hypothesis

The 080-088 specs varied **compute** (NL count) and **shape**
(visit distribution within {3..7}). 089 tests a third axis:
**band position** at fixed compute and fixed visit count.

Pattern body: `2,3,4,2,3,4,2,3,4,5,6,7` — 12 visits.
Combined with pre `[0,1]` + post `[8,9,10]`: **17 total layer-passes**.

Layer visits:

| Layer | 060A canonical | 089 shifted | Δ |
|---|---|---|---|
| 0,1 | 1 each | 1 each | 0 |
| **2** | **1** | **3** | **+2** (now in loop band) |
| 3 | 3 | 3 | 0 |
| 4 | 3 | 3 | 0 |
| **5** | **3** | **1** | **−2** (now post-loop) |
| 6,7 | 1 each | 1 each | 0 |
| 8,9,10 | 1 each | 1 each | 0 |

Net: **swap layer 2 into the loop band; layer 5 out.** The "loop
recurrence band" effectively shifts from {3,4,5} to {2,3,4} — one
layer earlier in the network depth.

**Mechanistic theory:** layer position in a transformer roughly
corresponds to feature abstraction depth. Layers 2-3 are typically
"shallow features" while 4-5 are "deeper composition." The canonical
{3,4,5} band sits at the shallow→deep transition. 089 tests whether
shifting the recurrence to *earlier* (more on shallow features) helps
or hurts.

Memory: #1726 tested wider bands like {2..7} (catastrophic, +0.163 bpb)
but did NOT test {2,3,4} specifically with NL=2 contiguous. So this
is an empirical hole.

## Baseline

Same as 080/081/082/085/087: 060A pre-quant post-EMA val_bpb at
canonical NL=2 (~1.06358).

## Expected Δ vs 060A

| Outcome | Δ bpb | Likelihood |
|---|---|---|
| Best (earlier band integrates shallow features better) | -0.0005 to +0.0005 | low-medium |
| Likely (band-position-sensitive; shift hurts) | +0.0010 to +0.0030 | medium-high |
| Worst (layer 2 wasn't trained for recurrence; destabilizes) | +0.0030 to +0.0080 | medium |

Higher likelihood of "loses" than 085/087 because layer 2 was trained
to be visited ONCE, not THREE times. The model's layer-2 weights are
optimized for non-recurrent processing.

## Accept criteria

- **Win:** pre-quant val_bpb ≤ **1.06330**
- **Noise:** 1.06330 – 1.06430
- **Kill:** ≥ 1.06430

This spec is **diagnostic-leaning** — most likely outcome is "loses
moderately, confirms canonical band is positionally optimal." That's
still informative.

## Config diff vs 060A baseline

```
LOOP_PATTERN = "2,3,4,2,3,4,2,3,4,5,6,7"
NUM_LOOPS    = 2
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.0   # eval-only
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
```

Index lists (verified):
- `encoder_indices = [0, 1, 2, 3, 4, 2, 3, 4]` (8 indices)
- `decoder_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-088.

### Compile-graph audit

Eval-only run. Model construction reads `LOOP_PATTERN` once at
`GPT.__init__`, building 17-element index lists baked-in. The
iteration order differs from canonical (and from prior 080-088
specs) — but the *number* of iterations and the *types* of layers
visited are the same as canonical.

Compiled `forward_logits` traces once on first eval batch with the
new iteration order. One new graph variant. From the second eval
batch onward, graph is cached. **No mid-run recompile.**

Bank weights load from 060A checkpoint via `load_state_dict`. Layer 2's
bank — trained for single-visit feedforward — is now read three
times per forward. Each read uses the same loaded weights. **No
weight slicing in compile.** **No narrow-K matmul.** **No dynamic
shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.** No TTT, no GPTQ.
- Wall: ~10 min.

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/089-eval-band-shift-234/seed_42/`.

Key artifacts:
- `train.log` — eval log
- `final.json` — pre-quant val_bpb at the shifted band

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (band shift severely broke model)
- Compile time > 6 min → kill
- NaN → kill

## Cost estimate

~$1 (eval-only, ~10 min on 4×H100).

## Reading 089 alongside 085/087/060A (matched-compute pattern axis)

| Spec | Compute | Mechanism | Loop band |
|---|---|---|---|
| 060A | 17 | canonical (3,3,3) | {3,4,5} |
| 085 | 17 | broad (2,2,3,2,2) | {3..7} |
| 087 | 17 | stepped (2,3,3,2) | {3..6} |
| **089** | **17** | **shifted (3,3,3)** | **{2,3,4}** |

Reading 089 alongside the others maps how the trained representation
responds to: (a) shape changes within the canonical band, (b) band
extension deeper, (c) band shift earlier. If 089 catastrophically
fails, the canonical band is positionally tuned. If 089 lands close
to 060A, recurrence transferability extends to layer 2 too —
genuinely new finding.

## Followup specs gated on result

- If 089 wins or near-noise: GG2 spec candidate (band shift to
  {4,5,6} later) becomes interesting — does direction matter?
- If 089 lands clearly worse than 085/087 by >0.001: layer-2 in the
  recurrence is specifically harmful; matches the canonical-band
  positional finding.
- If 089 ≈ 085/087: the band can shift positionally with similar
  effect to other shape changes — opens a 2D (position × shape) sweep.

## See also

- `research/specs/085-eval-time-broad-pattern-on-060A.md` — broad shape
- `research/specs/087-eval-time-stepped-pattern-on-060A.md` — stepped shape
- `runs/041H-loop45-frac035-screen/` — related to {4,5} band (training-time)
- `research/timelines/loop-recurrence-2026-04-27.md` — full historical timeline
- `research/ideas/parallelize-deep-looks.md` cluster GG (W8) and W10 decisions
