# Spec 100 — Eval-time compound NL=3 eq + 4 TTT phases on 060A (first compound spec)

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W21)
**Branch:** `exp/071-loop-pattern` (same as 080-099)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster TTT1 (W21).

## Hypothesis

**First compound spec.** After 20 single-axis specs (080-099)
exploring eval-time levers individually, this spec combines two of
the most likely winning levers:

1. **NL=3 equivalent recurrence** (from spec 080): 4 passes through
   {3,4,5} via `LOOP_PATTERN`. Spends the eval-time-headroom on
   deeper iteration.
2. **PHASED_TTT_NUM_PHASES=4** (from spec 091): adds a 4th TTT
   phase. Spends eval-time-headroom on more LoRA adaptation cycles.

Both single-axis specs use the eval-time-headroom for different
purposes. Compound test asks: **do they compose, or do they compete
for the same budget?**

Three reading scenarios:
1. **Constructive composition (additive):** 100 wins by approximately
   the sum of 080's gain and 091's gain (over canonical). Both
   levers exploit independent value.
2. **Sub-additive (competing):** 100 wins by less than the sum.
   The two levers partially overlap in what they exploit.
3. **Destructive composition:** 100 worse than either alone. The
   eval-time-headroom is consumed and one lever distorts the other.

This is the **first compound spec** in the 080+ series. Its result
informs whether further compounds (TTT2/TTT3 from W21 brainstorm)
are worth running.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

Tighter prediction than single-axis since this is gated on 080 and
091 individually winning. If both win:

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (additive) | -0.0010 to -0.0030 | low (depends on 080+091) |
| Plausible (sub-additive) | -0.0005 to -0.0010 | medium |
| Neutral (one helps, other neutral) | -0.0002 to +0.0005 | medium-high |
| Destructive | +0.0005 to +0.0020 | low-medium |

If 080 ≈ canonical or 091 ≈ canonical: 100 unlikely to win since
both single-levers must contribute.

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0010**
- **Win-conditional-on-both:** if 080 wins by Δ_a and 091 wins by
  Δ_b, 100 should land at canonical + α(Δ_a + Δ_b) for some α >
  0.5 (sub-additive at worst).
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
PHASED_TTT_NUM_PHASES = 4    # ← LEVER 2 (canonical 3, same as 091)
PHASED_TTT_PREFIX_DOCS = 2500
TTT_LORA_RANK = 80
TTT_LORA_LR = 1e-4
TTT_BETA1 = 0.0              # canonical
TTT_BETA2 = 0.99             # canonical
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

Index lists from 080:
- `encoder_indices = [0, 1, 2, 3, 4, 5, 3, 4, 5]` (9 indices)
- `decoder_indices = [3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10]` (11 indices)
- 20 total layer-passes (vs canonical 17).

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-098, plus the existing
`PHASED_TTT_NUM_PHASES` parameter.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. Two distinct compiled
functions (forward_logits, forward_ttt) each compile once on first
respective invocation with the NL=3-eq index lists. Both produce
one new graph variant each (same as 083's compile state). From the
second invocation of each onward, no recompile. **No mid-run
recompile.**

`PHASED_TTT_NUM_PHASES=4` is a Python loop bound; the compiled
forward_ttt graph is called one extra time per eval. **No new graph
variant from this lever.**

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Bank weights load from 060A checkpoint via `load_state_dict`.
Layer-indexed reads. Layers 3,4,5 each visited 4 times — bank rows
reused without modification. **No weight slicing.** **No narrow-K
matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~30-35 min total. Adds NL=3-eq overhead AND extra TTT phase:
  - Compile bursts (forward_logits + forward_ttt with new shapes):
    ~6-8 min combined
  - Pre-quant eval at NL=3 eq: ~7 min (vs canonical 5)
  - GPTQ calibration + quantization: ~3-5 min
  - Post-quant eval + 4 TTT phases: ~7-9 min (extra phase)

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/100-eval-NL3eq-4phases/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.075 → kill
- Compile time on either compiled function > 8 min → kill
- NaN → kill
- Total wallclock > 40 min → kill

## Cost estimate

~$4-5 (full pipeline eval, ~30-35 min on 4×H100).

## Sequencing

**Recommended:** run 080 and 091 first (each ~$1-4). If both win
(or 080 + 091 together suggest composition), launch 100. If either
loses substantially, abort 100.

## Reading 100 alongside 080 + 091 + canonical

Build the composition table:

| Spec | NL eq | TTT phases | Δ vs canonical (predicted) |
|---|---|---|---|
| 060A | 2 | 3 | 0 (reference) |
| 080 | 3 | 0 (TTT-off) | Δ_080 (TBD) |
| 091 | 2 | 4 | Δ_091 (TBD) |
| 083 | 3 | 3 (canonical TTT) | Δ_083 (TBD) |
| **100** | **3** | **4** | **Δ_100 (predicted: ≈ Δ_083 + Δ_091)** |

Reads as: does extending TTT on top of 083's deeper-recurrence-with-
TTT compose? If yes, 100 < 083. If no, 100 ≈ 083.

## Followup specs gated on result

- If 100 wins by more than 083 alone: composition works; spec 101
  candidate = TTT2 = NL=4 eq + 4 TTT phases (push further).
- If 100 ≈ 083: TTT-extension alone helps but doesn't compose with
  deeper recurrence (overlapping benefit).
- If 100 worse than 083: destructive composition; abort compound
  thread.

## See also

- `research/specs/080-eval-time-deeper-loop-on-060A.md` — single-lever sibling
- `research/specs/091-eval-time-more-TTT-phases-on-060A.md` — single-lever sibling
- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — canonical-TTT
  version (3 phases) of the deeper recurrence
- `research/ideas/parallelize-deep-looks.md` cluster TTT (W21)
