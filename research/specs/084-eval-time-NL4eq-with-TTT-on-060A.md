# Spec 084 — Eval-time NL=4 equivalent with TTT-on on 060A checkpoint

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint. No new code beyond 080-083 reuse.

**Date:** 2026-04-29 (autonomous wake W5)
**Branch:** `exp/071-loop-pattern` (same as 080/081/082/083)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: `parallelize-deep-looks.md`
cluster P1 (W3) extended.

## Hypothesis

Direct extension of spec 083 (TTT-on × NL=3 eq) to NL=4. Forms with
083 a 2-point TTT-on scaling curve at the upper end of the
recurrence-depth space:

| Spec | NL eq | TTT | GPTQ | Position on curve |
|---|---|---|---|---|
| 060A canonical | 2 | on | on | reference |
| 083 | 3 | on | on | one extra pass |
| **084** | **4** | **on** | **on** | **two extra passes** |

Reading 083 + 084 together answers two questions:
- Does deeper recurrence at eval scale with depth under TTT? If 084 <
  083 < canonical, the lever has slope and we should keep climbing.
- Is the gain monotone or does it saturate? If 084 ≈ 083, we found
  the practical limit on deeper-eval-recurrence.

This is a **conditional information** spec: most useful if 083
produces a measurable signal (gain or loss). If 083 is exactly noise,
084 has lower marginal value but still completes the curve.

## Baseline

Same as 083: 060A canonical post-TTT post-quant val_bpb (the real
leaderboard number for 060A seed 42).

## Expected Δ vs 060A canonical

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (recurrence scales with depth under TTT) | -0.0005 to -0.0030 | low |
| Plateau (083 captured the gain; 084 ≈ 083) | -0.0001 to +0.0005 | medium |
| Saturation regression (over-iteration after TTT) | +0.0005 to +0.0020 | medium |
| Destructive (TTT distorted at NL=4) | +0.0010 to +0.0040 | low-medium |

The saturation/regression scenarios are more likely at NL=4 than at
NL=3 (per Two-Scale Latent Dynamics: late-pass updates approach zero;
extra passes add noise more than signal).

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Win-conditional-on-083:** if 083 also wins, 084 should win by
  more than 083 (or at least equal) for "monotone improvement"
  conclusion.
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0010

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,6,7"  # NL=4 eq (same as 081)
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

Pattern body: 19 visits across layers 1-7 (same as spec 081).
Total: 23 layer-passes per sequence. Layers 3,4,5 each visited 5 times
(NL=4 equivalent).

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080/081/082/083.

### Compile-graph audit

Eval-only run, full pipeline (TTT + GPTQ). Same compile-graph story as
083, with one difference: the iteration count is 23 not 20.

Two distinct compiled functions:
1. **`forward_logits`** traces once on first eval batch with the
   23-iteration graph. One variant. Cached.
2. **`forward_ttt`** traces once on first TTT phase with the 23-
   iteration graph (different compiled function but same iteration
   count as forward_logits this run). One variant. Cached.

From the second invocation of each onward, no recompile.

The longer iteration count means each compile burst takes slightly
longer than 083 (~10-20% more compile time per function). Total
~40-60 sec extra compile time vs 083. **No mid-run recompile.**

Bank weights load identically to 083 (layer-indexed). **No weight
slicing in compile.** **No narrow-K matmul.** **No dynamic shapes.**

LoRA weight shapes determined at __init__ by `TTT_LORA_RANK=80`,
unchanged from 060A defaults.

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.**
- Wall: ~30-35 min total. Breakdown:
  - Compile burst 1 (forward_logits): ~4-5 min
  - Compile burst 2 (forward_ttt): ~4-5 min
  - Pre-quant eval at NL=4: ~9-10 min (vs canonical 5 min — 80% slower
    from 23 vs 17 layer-passes per sequence)
  - GPTQ calibration + quantization: ~3-5 min
  - Post-quant eval + TTT 3 phases: ~6-8 min

Total expected: 30-35 min. Cost: ~$4-5.

## Seed plan

1 seed (42) — matches 060A.

## Inputs / Outputs

- Same as 083. Output dir: `/workspace/runs/084-eval-NL4eq-TTT/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log
- `final_model.int6.ptz` — submittable artifact at NL=4 eval
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (deeper recurrence broke model)
- Post-TTT val_bpb > 1.075 → kill (TTT broke at NL=4)
- Compile time on either compiled function > 8 min → kill
- NaN → kill
- Total wallclock > 40 min → kill (budget exceeded)

## Cost estimate

~$4-5 (full pipeline eval, ~30-35 min on 4×H100).

## Sequencing with 083

**Recommended order:** run 083 FIRST. If 083 lands in the "destructive"
or "kill" range, **abort 084 launch** — TTT is incompatible with
deeper recurrence at NL=3, and NL=4 will only be worse. Save the $4-5.

If 083 lands in win or plausible band, run 084 to map the curve.

## Reading 083 + 084 + 060A together

Build the TTT-on scaling curve:

| NL eq | Spec | Layer-passes | Expected post-TTT bpb |
|---|---|---|---|
| 2 | 060A | 17 | (baseline) |
| 3 | 083 | 20 | conditional on 080 result |
| 4 | 084 | 23 | extension; saturation expected |

If post-TTT bpb is monotone-decreasing in NL: ship a multi-seed run
at the optimum. If U-shaped with minimum at NL=2: trained NL is
already optimal post-TTT. If monotone-increasing: deeper eval
recurrence is destructive under TTT.

## Followup specs gated on result

- If 084 < 083 by more than 0.0005: 085 candidate = TTT-on × NL=5 eq
  (keep climbing).
- If 084 ≈ 083: optimal eval-NL is ~3. Ship 083-promo at 8H × 600s.
- If 084 > 083: saturation hit at NL=3. Ship 083 if it won.

## See also

- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — direct sibling
- `research/specs/081-eval-time-NL4eq-on-060A.md` — TTT-off NL=4 sibling
- `research/ideas/parallelize-deep-looks.md` cluster P (W3) and W5 sequencing notes
