# Spec 250B — Eval-only PHASED_TTT_PREFIX_DOCS 2500 → 2750 on spec 250 outputs

**Date:** 2026-05-01 (deadline window)
**Idea:** TTT-side budget reinvestment, family member B.
**Lineage:** Spec 250 outputs (V21+slope0.3) + single env var override at eval
time. **Eval-only — no retrain.**

## Hypothesis

Spec 250's reported `total_eval_time` is `~514s` against the 600s cap, leaving
~86s of headroom inside the leaderboard-counted TTT eval budget. PR #1925
explored `PHASED_TTT_PREFIX_DOCS=3500` on PR #1855 base (essentially neutral),
but that's a different stack. V21's TTT regime (`TTT_MASK=no_qv`,
`TTT_LR_MULT=0.75`, `EVAL_SEQ_LEN=2560`) consumes less compute per chunk,
so longer prefix may be absorbable AND add signal here that 1925 didn't see.

This spec tests the **smallest meaningful prefix bump** (2500 → 2750, +10%).
If even this washes, larger jumps (3000+) won't help and we close the lever.
If it shows signal, 250D (= +500 doc bump to 3000) becomes a follow-on.

## Baseline

**Spec 250 same-seed post-TTT BPB.** Per-seed paired comparison. Reference:
spec 250 single-seed s42 = 1.05797 (from observed run); 3-seed mean target
1.0578 ± 0.0006.

## Expected Δ vs spec 250 (same seed)

**0 to −0.0003**, low confidence.

- For: more prefix docs = more TTT-time conditioning before scoring tail
  documents. V21's no_qv mask reduces per-chunk LoRA compute, freeing
  capacity that longer prefix can occupy.
- Against: TTT's effective signal saturates as prefix grows (diminishing
  returns); past some threshold, more docs add noise without information.
  PR #1925's neutral result on #1855 base at PREFIX=3500 suggests that
  base saturates somewhere between 2500-3500.

Author estimate of P(win ≥ −0.0002 same-seed vs 250): **~30-40%**.

## Accept criteria

- **Per-seed:** post-TTT BPB at PREFIX_DOCS=2750 ≥ −0.0002 below same-seed
  spec-250 result.
- **Mean across 3 seeds:** if mean Δ ≤ −0.0003, **promote** — submit 250B.
  If 0 ≥ mean Δ > −0.0003, **wash** — submit 250 unchanged. If mean Δ > 0,
  **kill**.
- All seeds clear 600s eval cap and 16 MB artifact cap.

## Config diff (vs spec 250)

Single env var override:

```
PHASED_TTT_PREFIX_DOCS = 2750     # was 2500
```

All other env vars identical to spec 250's launch script.

Add for eval-only execution:
- `RESUME_FROM_CKPT=/path/to/runs/250-1953-leakyrelu03/seed_<N>/final_model.pt`
- `TTT_EVAL_ONLY=1` (existing flag in train_gpt.py at line 3866 — skips
  training + GPTQ + diagnostic eval)

## Code changes

**None.** Same train code as spec 250 (`exp/250-1953-leakyrelu03` @ pinned
commit). No `exp/<slug>` branch needed. This spec lives on `research`.

## Hardware ladder

- **Mini:** SKIP. Eval-only single env override. Pattern validated by
  existing `TTT_EVAL_ONLY=1` flag.
- **Official (8×H100):** seeds matching spec 250's seeds (42, 0, 1234),
  **eval-only**. Each seed re-runs only the TTT phased eval pipeline.
  ~10-12 min wallclock per seed.

## Seed plan

- **Seed 42 first** as smoke-and-budget-check. Verify post-TTT eval_time
  ≤ 580s on this seed (50s buffer below 600s cap).
- If seed 42 paired Δ ≥ +0.0005 OR eval_time > 595s, **abort 0 and 1234**.
- Otherwise launch 0 and 1234 sequentially.

## Inputs

- **Required:** spec 250 must have completed and produced
  `runs/250-1953-leakyrelu03/seed_{42,0,1234}/final_model.{pt,int6.ptz}`.
- Tokenizer + data: same as spec 250.

## Checkpoints to emit

- Per-seed `submission.json` matching spec 250's schema.
- Per-seed `train.log` (eval-only — TTT phased eval lines + final val_bpb).

Destination: `runs/250B-prefix-docs-2750/seed_{42,0,1234}/`.

## Stop-early criteria

- **Per-seed:**
  - Eval-only run errors during model load → kill seed.
  - Eval wallclock > 595s during the TTT phased eval → kill seed (cap
    breach imminent).
  - Post-TTT BPB sanity floor: > 1.080 → kill seed.
- **Across seeds:** if seed 42 paired Δ ≥ +0.0005 or eval_time > 595s,
  abort remaining seeds.

## Cost estimate

- ~$3-5 per seed re-eval (8×H100, eval-only, ~12 min/seed).
- 3 seeds total: **~$9-15**.

## Extra artifacts

- Per-seed `paired_delta.txt`: `seed=<N> spec250_bpb=<X> spec250B_bpb=<Y> delta=<Y-X>
  spec250B_eval_time_s=<T>`.
- Mean Δ summary at end + max eval_time across seeds.

## Open questions for interview

1. **Spec 250 status.** Hard dependency.
2. **Pod region & capacity.** NE-1 first, JP fallback, 8×H100 only.
3. **Pod re-use** between 250 / 250B / 250C arms.
4. **Failure halting** if seed 42 cap-breaches or regresses.
5. **Submission decision.** If both 250B and 250C win, does 250B (smaller
   change) take priority for submission, or do we run a stack as 250F?
   Default: pick lower 3-seed mean.

## Notes

- Family member of 250BCE (eval-only TTT-side sweeps on spec 250 outputs).
- Cheapest expected gain in the family. If this washes, larger PREFIX_DOCS
  values (3000) won't help.
- Strictly downstream of spec 250.
- Do not stack 250B with 250C in this round — independent attribution
  needed first.
