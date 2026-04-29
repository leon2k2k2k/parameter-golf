# Spec 083 — Eval-time NL=3 equivalent with TTT-on on 060A checkpoint

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint. No new code beyond what 080 uses.

**Date:** 2026-04-29 (autonomous wake W4)
**Branch:** `exp/071-loop-pattern` (same as 080/081/082)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: `parallelize-deep-looks.md`
cluster P1 (W3) — composes 080 with TTT.

## Hypothesis

Spec 080 measured pre-quant val_bpb at NL=3 equivalent **with TTT off**.
That isolates the deeper-recurrence effect from TTT effects. But the
real leaderboard number includes Phased TTT (3 phases × LoRA updates
adapting to the eval distribution).

This spec turns TTT back on and runs eval at NL=3 equivalent. Tests
**whether the deeper-recurrence gain (if any) composes with TTT, or
whether TTT was already capturing the gain that deeper recurrence
would otherwise provide**.

Three reading scenarios:
1. **Compose constructively (additive):** 083 < 080 by ~the TTT-baseline-
   gain. Both levers help; ship 083 to leaderboard.
2. **Compose destructively (TTT-already-captures-it):** 083 ≈ canonical
   TTT-on bpb (no extra deeper-recurrence gain after TTT). The TTT LoRA
   was already encoding "more refinement"; adding NL=3 is redundant.
3. **TTT distorted by deeper recurrence:** 083 > 080 at TTT-off-comparison
   point. The TTT phases were tuned for NL=2 dynamics; running them at
   NL=3 destabilizes the LoRA adaptation.

The leaderboard interpretation is straightforward: if 083 < canonical-
TTT-on bpb, ship it.

## Baseline

Two reference points:
- **060A canonical** (NL=2, TTT-on, GPTQ-on, full pipeline):
  pre-quant ~1.06358, post-quant ~1.0540s post-TTT (per memory
  `project_baseline_1851_post_1872`).
- **Spec 080** (NL=3 eq, TTT-off): pending result.

The compare-of-interest is 083 vs 060A's canonical TTT-on result on
the same seed.

## Expected Δ vs 060A canonical-TTT-on

| Outcome | Δ pre-quant val_bpb | Δ post-TTT val_bpb | Likelihood |
|---|---|---|---|
| Best (composes additive) | -0.0005 to -0.0020 | -0.0005 to -0.0030 | low-medium |
| Plausible (TTT captures most) | -0.0001 to +0.0005 | ≈ 0 | medium |
| Destructive (TTT distorted) | +0.0005 to +0.0020 | +0.0010 to +0.0040 | low-medium |

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0010

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,3,4,5,3,4,5,6,7"  # same as 080: NL=3 eq
NUM_LOOPS    = 2
ENABLE_LOOPING_AT = 0.0      # eval-only, immediately active
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 1             # diff from 080
PHASED_TTT_ENABLED = 3       # diff from 080
PHASED_TTT_NUM_PHASES = 3
PHASED_TTT_PREFIX_DOCS = 2500
GPTQ_CALIBRATION_BATCHES = 16  # diff from 080 — quant on for full pipeline
```

(All other env vars per 060A defaults.)

## Code changes

**None.** Reuses LOOP_PATTERN env var introduced at `e7ccda2`, plus
existing TTT and GPTQ infrastructure on the 060A code base.

### Compile-graph audit

Eval-only run with TTT phases. Two distinct compiled functions:

1. **`forward_logits`** (used for scoring): traces once on first eval
   batch with the longer iteration count from LOOP_PATTERN. One graph
   variant. Cached.
2. **`forward_ttt`** (used inside TTT phases for LoRA-applied scoring):
   traces once on first TTT phase with the same longer iteration
   count. Different compiled function from forward_logits but same
   pattern length. One graph variant. Cached.

Both are pre-compiled at first respective invocation. From the second
batch / second TTT phase onward, no recompile. **No mid-run recompile.**

The TTT LoRA path uses `_block_with_lora` which wraps each Block call
with a low-rank adapter applied to attention/MLP banks. The LoRA
shapes are determined at __init__ time by `TTT_LORA_RANK=80`,
unchanged. **No weight slicing, no narrow-K matmul, no dynamic
shapes.**

Memory `feedback_loop_activation_compile_is_cache_not_recompile`
confirms the first-call compile-burst pattern is cache production,
not mid-run recompile.

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.**
- Wall: ~25-30 min total. Breakdown:
  - Compile burst 1 (forward_logits): ~3-4 min
  - Compile burst 2 (forward_ttt): ~3-4 min
  - Pre-quant eval at NL=3: ~7 min (vs canonical 5 min, +30% from
    extra passes)
  - GPTQ calibration + quantization: ~3-5 min
  - Post-quant eval + TTT 3 phases: ~5-7 min

Total expected: 25-30 min. Cost: ~$3-4.

## Seed plan

1 seed (42) — matches 060A.

## Inputs / Outputs

- Same as 080. Output dir: `/workspace/runs/083-eval-NL3eq-TTT/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log including TTT phases
- `final_model.int6.ptz` — submittable artifact at NL=3 eval
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (deeper recurrence broke model)
- Post-TTT val_bpb > 1.075 → kill (TTT broke at deeper recurrence)
- Compile time on either compiled function > 6 min → kill
- NaN → kill
- Total wallclock > 35 min → kill (budget exceeded)

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Open questions

1. **Wallclock fit:** with NL=3 adding ~30% to eval time, total may
   approach the 600s scoring budget on 8H but on 4H matched-FLOPs the
   eval budget is doubled (1200s wallclock target). Should fit.
2. **Submittable?** If 083 wins, the resulting `.int6.ptz` is a
   leaderboard-valid artifact ONLY if total scoring time is ≤ 600s on
   8H. Need to verify by promoting to 8H run as 083-promo.
3. **TTT prefix at deeper NL.** PHASED_TTT_PREFIX_DOCS=2500 was tuned
   at NL=2. Deeper recurrence may benefit from longer prefixes (more
   adaptation budget) or shorter (less LoRA overfit). Default to 2500
   as the canonical setting; sweep candidate for followup.

## Reading 083 alongside 080

| Spec | TTT | NL eq | What it tells us |
|---|---|---|---|
| 060A canonical | on | 2 | reference: full leaderboard pipeline at trained NL |
| 080 | off | 3 | does deeper recurrence change pre-quant bpb? |
| **083** | **on** | **3** | **does 080's signal survive TTT + GPTQ → leaderboard?** |

If 080 wins and 083 wins by similar magnitude → strong stack. Promote
to multi-seed.

If 080 wins but 083 ≈ canonical → TTT was already capturing the gain.
Either way the homestretch outcome is "TTT-on submission stays at
canonical NL." No new submission lever.

If 080 ≈ canonical but 083 wins → unlikely but interesting; would
indicate TTT specifically benefits from deeper recurrence at adaptation
time.

## Followup specs gated on result

- If 083 wins (post-TTT < canonical post-TTT − 0.0005):
  - **083-promo** at 8×H100 leaderboard run, 1 seed first then 3-seed
    if confirmed.
  - **084 candidate:** TTT-on × NL=4 eq (composes 081 with TTT).
- If 083 ≈ canonical: TTT captures the gain; pivot away from this axis.
- If 083 < canonical (loses): TTT incompatible with deeper eval-time
  recurrence; the eval-time-headroom lever closes.

## See also

- `research/specs/080-eval-time-deeper-loop-on-060A.md` — TTT-off sibling
- `research/specs/081-eval-time-NL4eq-on-060A.md`
- `research/specs/082-eval-time-NL1eq-on-060A.md`
- `research/ideas/parallelize-deep-looks.md` cluster P (W3) and W
  (W4 — post-quant variants)
