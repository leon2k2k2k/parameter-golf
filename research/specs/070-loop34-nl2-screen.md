# Spec 070 — Loop band {3,4} NL=2 on 060A baseline (4H 20-min screen)

**Status:** FROZEN — config-only on 060A pinned commit. Ready to run.

**Date:** 2026-04-29
**Branch:** none (config-only)
**Pinned commit:** `a0a48b7` (head of `exp/060-resume-ckpt`, the 060A baseline)
**Parent:** 060A (#1855 port).

## Hypothesis

The 041 series tested loop-band variations exhaustively on the *shrunk*
{4,5} band but never tested **{3,4}**. At matched layer-pass count
(17 each), canonical {3,4,5}NL=2 (1.06514) beats {4,5}NL=3 (1.06563)
by **0.00050 bpb** — meaning layer 3 in the band has a small but
measurable contribution.

**Hypothesis:** layer 3 is the load-bearing loop visit. If true, then
**{3,4}NL=2 should beat {4,5}NL=2** (both 15-pass configs):
- 041H — {4,5}NL=2, frac=0.35 — landed at 1.06693
- 070 — {3,4}NL=2, frac=0.35 — predicted ~1.06600 to 1.06650

And vs canonical {3,4,5}NL=2 (17-pass), 070 has 15 passes per step =
**12% lower step-time = ~12% more steps in the same wallclock**. The
extra steps may compensate for the small quality cost of dropping
layer 5 from the loop, landing at parity or modest win.

**Mechanistic theory:** layer 3 sits at the boundary between the
shallow encoder and the deep representation band. It also is the
entry to the U-Net recurrence (`encoder_indices` starts the loop
band there). Looping it iterates the integration of low-level
features into the deep representation; looping {4,5} just refines
an already-formed representation — diminishing marginal value.

This is a **principled-not-just-portering test**: predicts a
specific outcome from a mechanism, not "try one more lever."

## Baseline

060A single-seed (1.06438 pre-quant EMA, run on 4×H100 matched-FLOPs
config — see `runs/060A-1855-port/`).

## Expected Δ vs 060A

| Outcome | bpb | Likelihood |
|---|---|---|
| Best (layer 3 carries loop, steps compensate) | −0.0005 to 0 | low-medium |
| Likely (between 041H and canonical) | +0.0005 to +0.0015 | medium |
| Worst (layer 3 in band doesn't help; steps don't recover) | +0.0015 to +0.0030 | medium |

## Accept criteria

- **Win:** pre-quant EMA ≤ **1.06464** (within 0.00026 of 060A baseline)
- **Noise:** 1.06464 – 1.06614
- **Kill:** ≥ 1.06614

If 070 lands in win or noise band, 070b followup tests
`ENABLE_LOOPING_AT=0.25` (the 041I early-activation lever).

## Config diff vs 060A

```
LOOP_START = 3   (unchanged)
LOOP_END   = 4   (was 5)
NUM_LOOPS  = 2   (unchanged)
ENABLE_LOOPING_AT = 0.35   (unchanged — keeps timing isolated for clean comparison)
```

All other env vars verbatim from 060A.

## Code changes

**None.** Pure env-var override on 060A pinned commit.

## Hardware ladder

- **Single rung: 4×H100, 1 seed (42), 20-min wallclock (1200s).**
- Matched-FLOPs to 8H × 600s via `GRAD_ACCUM_STEPS=2`.
- No mini rung (config-only change; smoke unnecessary).
- **Result is NOT leaderboard-valid** (training >600s on 4H rung). For
  research-side band-comparison only. If 070 wins, promote to 8H × 600s
  as 070-promo for a leaderboard-valid measurement.

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths.

## Checkpoints emitted

- `final_model.pt` — pre-quant post-EMA
- `final_model.int6.ptz` — post-quant submission blob
- `train.log`

Saved to `/workspace/runs/070-loop34-nl2-screen/seed_42/`.

## Stop-early criteria

- train_loss > 5.0 at step 1000 → kill (training broken)
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- mid-run torch.compile recompile → kill (always-tensor violated; should not occur — config-only)
- NaN → kill
- Wallclock exceeds 1300s (hard cap, 100s buffer over budget)

## Cost estimate

~$2.50 (4×H100 × 1200s ≈ 20 min wallclock + ~5 min eval).

## Open questions for interview

1. **Region.** Standard NE-1 first, JP fallback per memory.
2. **Halt action** if step-count comes in 5%+ below 041H's measured
   (which had similar layer-pass count): re-check throughput per
   `feedback_pod_throughput_check`; possibly re-provision pod.
3. **Stop pod after?** Yes per default policy.

## Followups gated on result

- If win/noise: **070b** with `ENABLE_LOOPING_AT=0.25` to combine
  with 041I's early-activation lever (~$2.50 more).
- If win: **070-promo** at 8×H100 × 600s for leaderboard-valid
  measurement, possibly multi-seed (~$5-15).
- If kill: closes the {3,4}-band hypothesis. Documented as the third
  empirical pillar of the loop-band exploration (alongside 040
  no-loop and 041H {4,5}NL=2).

## Why this is worth running

- **Empirical hole:** {3,4} band has never been tested in our 30+ specs
  on the loop axis.
- **Cheap:** ~$2.50 single-shot.
- **Predicted outcome from a mechanism**, not just lever portering.
- **Information value high either way:** confirms or refutes "layer 3
  is the load-bearing visit," which informs every future loop-band
  spec.

## See also

- `research/specs/041L-loop45-nl3-frac025-screen.md` — 041 series
  combined-lever spec
- `research/specs/041N-loop45-nl4-frac035-screen.md` — NL scaling
  at fixed band
- `runs/040-no-loop-ablation-screen/` — no-loop baseline (1.07222)
- `runs/041H-loop45-frac035-screen/` — {4,5}NL=2 baseline (1.06693)
- `research/timelines/loop-recurrence-2026-04-27.md` — full
  recurrence-axis timeline
