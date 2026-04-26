---
name: Spec 045 arm invalidation — Lever A never trained
description: Arms A, AC, E, F in spec 045 all had LOOP_ITER_EMBEDS=1 as a silent no-op; iter embeds stayed at zero. Results must be reinterpreted.
type: project
---

`loop_iter_embeds` was not in any optimizer group (see feedback memory: GPT-root params). All arms using `LOOP_ITER_EMBEDS=1` had embeds frozen at zero init throughout training.

**Corrected interpretation of spec 045 arms:**

| Arm | Config | What actually ran | Result |
|---|---|---|---|
| A | `LOOP_ITER_EMBEDS=1` | Baseline (embeds = zero no-op) | 1.06576 (+0.00062, noise) |
| AC | `LOOP_ITER_EMBEDS=1 LOOP_SCALE_INIT=recip` | Just C (1/L init) | 1.06472 (−0.00042, ≈ D) |
| D | `LOOP_SCALE_INIT=recip` | Just C | pending |
| E | `NUM_LOOPS=3 LOOP_START=4 LOOP_END=5 LOOP_ITER_EMBEDS=1 LOOP_SCALE_INIT=recip` | Loop45 NL=3 + C only | queued |
| F | `NUM_LOOPS=3 LOOP_START=3 LOOP_END=5 LOOP_ITER_EMBEDS=1 LOOP_SCALE_INIT=recip` | Loop345 NL=3 + C only | queued |

The "AC synergy" observed in mid-run training loss (AC >> D) was pod variance noise, not real. AC ≈ D in final val_bpb.

**Fix:** Commit `fc54262` on `exp/045-loop-layer-improvements` adds `loop_iter_embeds` to `scalar_params` in `Optimizers.__init__`.

**New arms using fix (first real tests of Lever A):**
- Arm A2: `LOOP_ITER_EMBEDS=1` alone — first real test of iter embeds
- Arm G: `LOOP_ITER_EMBEDS=1 LOOP_SCALE_INIT=recip LOOP_LR_SCALE=recip`
- Arm H: `LOOP_ITER_EMBEDS=1 LOOP_SCALE_INIT=recip LOOP_PER_PASS_RESID_MIX=1`
- Arm GH: all four flags

**Why:** The only validated result from spec 045 so far is Lever C (1/L init). Everything else needs to be rerun on `fc54262`.
