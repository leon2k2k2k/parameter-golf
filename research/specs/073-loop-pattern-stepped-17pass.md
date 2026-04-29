# Spec 073 — Loop pattern 2-3-3-2 across {3,4,5,6} at matched-canonical 17 passes

**Status:** FROZEN — reuses code from `exp/071-loop-pattern @ e7ccda2`. Ready to run.

**Date:** 2026-04-29
**Branch:** `exp/071-loop-pattern` (shared with 071/072/074)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent:** 060A (#1855 port).

## Hypothesis

071/072 test 15-pass band-shape variants (~12% throughput recovery,
trades quality for steps). **This spec tests pattern shape at matched
compute** — same 17 layer-passes as canonical {3,4,5}NL=2, no
throughput delta. Isolates pattern-shape effect from compute effect.

Pattern body: `1,2,3,4,5,3,4,5,4,5,6,6,7,8` (14 visits).
Combined with pre `[0]` + post `[9,10]`: **17 total layer-passes**.

| Layer | 073 visits | Canonical {3,4,5}NL=2 visits | Δ |
|---|---|---|---|
| 0 | 1 | 1 | 0 |
| 1,2 | 1 each | 1 each | 0 |
| **3** | **2** | 3 | −1 |
| **4** | **3** | 3 | 0 |
| **5** | **3** | 3 | 0 |
| **6** | **2** | 1 | +1 |
| 7 | 1 | 1 | 0 |
| 8 | 1 | 1 | 0 |
| 9,10 | 1 each | 1 each | 0 |

Net effect: **layer-band {3,4,5,6} visit profile = 2,3,3,2** instead of
canonical's 3,3,3,1. Same compute (17 visits) — recurrence is *extended*
one layer deeper (into 6) at the cost of one visit at layer 3.

**Mechanistic theory:** if recurrence value benefits from extending
into adjacent depth (layer 6 brought into the loop), the 2-3-3-2
profile across {3,4,5,6} should beat the canonical 3-3-3 concentrated
on {3,4,5}. The trade is one extra visit at layer 6 for one fewer
at layer 3. If layer 6 contributes more in the recurrence than the
3rd visit to layer 3, this wins.

The encoder/decoder index lists:
- `enc=[0,1,2,3,4,5,3,4]` (8 indices, ascending then descent)
- `dec=[5,4,5,6,6,7,8,9,10]` (9 indices, alternating then ascending)

## Baseline

060A single-seed (1.06358 pre-quant EMA on 4×H100 matched-FLOPs).

## Expected Δ vs 060A

Tighter prediction range than 071/072 because compute is matched:

| Outcome | bpb | Likelihood |
|---|---|---|
| Best (depth concentration helps) | −0.0010 to 0 | low-medium |
| Likely (canonical is well-tuned, shape ~ neutral) | −0.0003 to +0.0015 | medium |
| Worst (staggering disrupts U-Net coherence) | +0.0015 to +0.0030 | medium-low |

## Accept criteria

- **Win:** pre-quant EMA ≤ **1.06384**
- **Noise:** 1.06384 – 1.06534
- **Kill:** ≥ 1.06534

## Config diff vs 060A

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,4,5,6,6,7,8"
NUM_LOOPS    = 2          # any positive value enables looping_active
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.35  # unchanged
```

## Code changes

Same `exp/071-loop-pattern @ e7ccda2` as 071/072/074. No additional changes.

Index list (verified):
- `encoder_indices = [0, 1, 2, 3, 4, 5, 3, 4]`
- `decoder_indices = [5, 4, 5, 6, 6, 7, 8, 9, 10]`
- 17 total visits.

## Hardware ladder

- **Mini rung: SKIP** — same code path as 071/072. Plumbing verified there.
- **Single rung: 4×H100, 1 seed (42), 20-min wallclock (1200s)**.

Result research-side only. Promote to 8H × 600s as 073-promo if win.

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths.

## Stop-early criteria

- train_loss > 5.0 at step 1000 → kill
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- mid-run torch.compile recompile → kill
- LOOP_PATTERN parse error → kill

## Cost estimate

~$2.50.

## Followups gated on result

- If win: 073-promo at 8H × 600s.
- 074 is sibling spec, broader spread (layer 7 also at 2 visits).
- If both 073/074 lose: matched-compute pattern shape doesn't help; canonical is genuinely tuned.

## See also

- `research/specs/074-loop-pattern-broad-17pass.md` — sibling, broader spread
- `research/specs/071-loop-pattern-tent-235.md` — 15-pass variant
- `research/specs/070-loop34-nl2-screen.md` — 15-pass band shrink (lost +0.0024)
