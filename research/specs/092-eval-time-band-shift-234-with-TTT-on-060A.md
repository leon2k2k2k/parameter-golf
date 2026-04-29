# Spec 092 — Eval-time band-shift {2,3,4} WITH TTT on 060A (leaderboard cell)

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W13)
**Branch:** `exp/071-loop-pattern` (same as 080-091)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster VV1 (W13).

## Hypothesis

Sibling to 089 (band-shift {2,3,4} eval, TTT off). 089 tested whether
shifting the loop band one layer earlier worked at the pre-quant
level. 092 enables TTT and GPTQ — completes the leaderboard cell for
the earlier-shift position.

Pattern body: `2,3,4,2,3,4,2,3,4,5,6,7` (same as 089). 17 total
layer-passes. Loop band shifted from canonical {3,4,5} to {2,3,4}.

Together with future spec 093 candidate (TTT-on band {4,5,6}), forms
the position × TTT grid:

| Band | TTT off | TTT on |
|---|---|---|
| {3,4,5} canonical | 060A | 060A canonical (reference) |
| {2,3,4} earlier | 089 | **092** |
| {4,5,6} later | 090 | 093 (W14 candidate) |

**Hypothesis evolution:** if 089 lost without TTT, can TTT recover
the loss? Two scenarios:
1. **TTT can adapt to non-canonical band:** 092 closes the gap or
   even surpasses canonical 060A.
2. **TTT amplifies the error:** 092 worse than 089 (because the
   LoRA adapts based on the wrong-band model's outputs, distorting
   adaptation).

The trained TTT LoRA has rank 80 — significant capacity to adapt.
But the underlying recurrence is OOD vs trained — TTT may struggle
to compensate for fundamentally OOD layer-2 recurrence.

## Baseline

Two reference points:
- 060A canonical post-TTT (the leaderboard number).
- Spec 089 result (089's pre-quant val_bpb).

Compare-of-interest: 092 vs 060A canonical post-TTT.

## Expected Δ vs 060A canonical post-TTT

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (TTT recovers most of 089's loss) | -0.0005 to +0.0005 | low |
| Plausible (TTT partial recovery) | +0.0005 to +0.0020 | medium |
| Likely (TTT cannot fix the OOD recurrence) | +0.0020 to +0.0050 | medium-high |
| Worst (TTT distorted) | +0.0050 to +0.0100 | medium |

Higher likelihood of "loses substantially" than 086/088 (which had
gentler shape changes).

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0020

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "2,3,4,2,3,4,2,3,4,5,6,7"   # same as 089
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

Index lists (verified earlier in 089):
- `encoder_indices = [0, 1, 2, 3, 4, 2, 3, 4]` (8 indices)
- `decoder_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-091.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. Two distinct compiled
functions (forward_logits, forward_ttt) each compile once on first
respective invocation with the new band-shifted index lists. Both
produce one new graph variant each (different from canonical and from
080-091 patterns). From the second invocation of each onward, no
recompile. **No mid-run recompile.**

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Bank weights load from 060A checkpoint. Layer 2's bank — visited
3 times in this pattern — reuses the loaded layer-2 weights without
modification on each visit. **No weight slicing in compile.** **No
narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min total. Same breakdown as 086/088 (matched compute):
  - Compile bursts (forward_logits + forward_ttt): ~6-8 min combined
  - Pre-quant eval: ~5-6 min
  - GPTQ + quantization: ~3-5 min
  - Post-quant eval + 3 TTT phases: ~5-7 min

Cost: ~$3-4.

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/092-eval-band-shift-234-TTT/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log
- `final_model.int6.ptz` — submittable artifact at the shifted band
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.080 → kill (band shift broken even after TTT)
- Compile time on either compiled function > 6 min → kill
- NaN → kill
- Total wallclock > 35 min → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

**Recommended order:**
1. Run 089 (TTT-off) first — ~$1, ~10 min. If 089 catastrophically
   loses, abort 092 (TTT can't fix fundamentally broken recurrence
   at the level of seeing the wrong band).
2. Run 092 if 089 was within plausible bounds.

## Reading 089 + 092 + canonical

Question: **does TTT compensate for non-canonical band positions?**

| Spec | Band | TTT | What it tells us |
|---|---|---|---|
| 060A | {3,4,5} | on | reference |
| 089 | {2,3,4} | off | "raw" earlier-shift impact (pre-TTT) |
| **092** | **{2,3,4}** | **on** | **does TTT recover 089's loss?** |

If 092 ≈ 089 (similar loss vs canonical): TTT doesn't help at this
band shift. The shift is fundamental.
If 092 < 089 by clear margin: TTT amortizes the shift. Closer-to-
canonical-than-pre-quant says TTT is a flexible adapter.
If 092 < canonical: shift+TTT actually wins (unlikely but interesting).

## Followup specs gated on result

- If 092 wins or near-noise: spec 093 candidate = TTT-on band {4,5,6}
  (completes the position × TTT grid).
- If 092 ≈ 089 (TTT doesn't help): position is fundamental;
  TTT-side cannot rescue band-shift.
- If 092 catastrophically worse than 089: TTT distorts at OOD bands.

## See also

- `research/specs/089-eval-time-band-shift-234-on-060A.md` — TTT-off sibling
- `research/specs/090-eval-time-band-shift-456-on-060A.md` — paired position
- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — TTT-on template
- `research/ideas/parallelize-deep-looks.md` cluster VV (W13)
