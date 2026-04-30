# Spec 250C — Eval-only PHASED_TTT_NUM_PHASES 3 → 4 on spec 250 outputs

**Date:** 2026-05-01 (deadline window)
**Idea:** TTT-side budget reinvestment, family member C.
**Lineage:** Spec 250 outputs (V21+slope0.3) + single env var override at eval
time. **Eval-only — no retrain.**

## Hypothesis

Spec 250's `total_eval_time` ~514s leaves ~86s of TTT-budget headroom.
Adding one more TTT phase (3 → 4) costs ~+65s — fits with ~21s safety
margin. Each TTT phase fits a fresh LoRA on a longer cumulative prefix;
one extra phase gives the late-document scoring more conditioning.

Spec 091 explored `NUM_PHASES=4` on 060A baseline. V21's TTT regime is
strictly stronger (no_qv mask, longer ctx, tuned LR), so transfer is
uncertain. This spec is the canonical V21 test of the lever.

## Baseline

**Spec 250 same-seed post-TTT BPB.** Per-seed paired comparison.
Reference: spec 250 single-seed s42 = 1.05797.

## Expected Δ vs spec 250 (same seed)

**0 to −0.0005**, low-medium confidence.

- For: an extra phase adds a fresh LoRA fit on a longer cumulative
  prefix. Mechanically each phase adapts to a chunk of the val stream;
  more phases = finer-grained adaptation. Strongest plausibility on
  longer prefixes (which 250B is testing) — composes naturally if both
  win.
- Against: phase boundaries currently set at doc-rank 833/1666/2500
  (3 phases over 2500 prefix). Re-bucketing into 625/1250/1875/2500
  changes the per-phase data quantum, potentially below the threshold
  where LoRA can fit usefully. Spec 091's 060A result (status not
  documented in my survey) — could go either way.
- Cap-safety: 21s margin before 600s cap is tight. One unlucky seed
  wallclock spike will breach.

Author estimate of P(win ≥ −0.0003 same-seed vs 250): **~25-35%**.

## Accept criteria

- **Per-seed:** post-TTT BPB at NUM_PHASES=4 ≥ −0.0003 below same-seed
  spec-250 result.
- **Mean across 3 seeds:** if mean Δ ≤ −0.0005, **promote** — submit 250C.
  If 0 ≥ mean Δ > −0.0005, **wash** — submit 250 unchanged. If mean Δ > 0,
  **kill**.
- **Cap safety:** all seeds must clear 600s eval cap with at least 5s
  margin. If any seed breaches 595s, that seed is disqualified for
  submission even if BPB is good.
- All seeds clear 16 MB artifact cap.

## Config diff (vs spec 250)

Single env var override:

```
PHASED_TTT_NUM_PHASES = 4     # was 3
```

All other env vars identical to spec 250's launch script. Note: phase
boundary positions may auto-recompute based on `PHASED_TTT_PREFIX_DOCS=2500`
divided by 4 = 625 docs per phase.

Add for eval-only execution:
- `RESUME_FROM_CKPT=/path/to/runs/250-1953-leakyrelu03/seed_<N>/final_model.pt`
- `TTT_EVAL_ONLY=1`

## Code changes

**None.** Same train code as spec 250.

## Hardware ladder

- **Mini:** SKIP. Eval-only single env override.
- **Official (8×H100):** seeds 42, 0, 1234. **Eval-only.**

## Seed plan

- **Seed 42 first** as smoke-and-budget-check. **Cap-safety verification:
  this seed's TTT eval_time is the primary screening criterion** for the
  lever. If seed 42 eval_time > 595s, abort 0 and 1234 even if BPB looks
  good — wallclock variance on subsequent seeds will breach the cap.
- If seed 42 paired Δ ≥ +0.0005 OR eval_time > 595s, **abort 0 and 1234**.
- Otherwise launch 0 and 1234 sequentially.

## Inputs

- **Required:** spec 250 outputs (same as 250B).

## Checkpoints to emit

- Per-seed `submission.json`.
- Per-seed `train.log`.

Destination: `runs/250C-num-phases-4/seed_{42,0,1234}/`.

## Stop-early criteria

- **Per-seed:**
  - Eval-only run errors during model load → kill seed.
  - Eval wallclock > 595s during the TTT phased eval → kill seed (cap
    breach imminent — this is the primary risk for this spec).
  - Post-TTT BPB sanity floor: > 1.080 → kill seed.
- **Across seeds:** if seed 42 paired Δ ≥ +0.0005 OR eval_time > 595s,
  abort 0 and 1234.

## Cost estimate

- ~$3-5 per seed (8×H100, eval-only, ~12-13 min/seed).
- 3 seeds total: **~$9-15**.

## Extra artifacts

- Per-seed `paired_delta.txt`: `seed=<N> spec250_bpb=<X> spec250C_bpb=<Y>
  delta=<Y-X> spec250C_eval_time_s=<T>`.
- **Eval-time max across seeds is critical** for this spec — needs to
  appear in summary.

## Open questions for interview

1. **Spec 250 status.** Hard dependency.
2. **Pod region & capacity.** NE-1 first, JP fallback, 8×H100 only.
3. **Cap-safety thresholds.** Default 595s as the cap-safety floor; if you
   want stricter (590s, more buffer) or looser (598s, more headroom), set
   before launch.
4. **Phase boundary recomputation.** If PHASED_TTT_NUM_PHASES is decoupled
   from PHASED_TTT_PREFIX_DOCS in the eval pipeline (i.e., phases at
   fixed positions like 833/1666/2500 regardless of count), this lever
   does something different than my hypothesis. Execution should verify
   in seed 42's train.log that the phase split is `prefix_docs / num_phases`.
5. **Submission decision.** Same as 250B: if multiple arms win, pick
   lower 3-seed mean.

## Notes

- Family member of 250BCE (eval-only TTT-side sweeps on spec 250 outputs).
- Higher cap-safety risk than 250B (21s margin vs 36s margin).
- If 250B and 250C both win independently, **a stacked spec 250F**
  (PREFIX_DOCS=2750 + NUM_PHASES=4) would be the natural follow-on
  (~+115s combined, breaches 600s cap — would need more aggressive
  trimming elsewhere or larger gains to justify).
- Strictly downstream of spec 250.
