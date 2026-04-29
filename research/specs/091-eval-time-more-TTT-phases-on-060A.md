# Spec 091 — Eval-time more TTT phases (4 instead of 3) at canonical NL on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W12)
**Branch:** `exp/071-loop-pattern` (same as 080-090; LOOP_PATTERN unset)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster SS1 (W12).

## Hypothesis

The 080-090 specs spent eval-time-headroom on **extra layer-passes**
(LOOP_PATTERN deeper or shape-shifted). 091 spends it on a **different
eval-time lever**: more TTT phases.

Default `PHASED_TTT_NUM_PHASES=3`. Each phase performs:
1. LoRA update on prefix-doc tokens (TTT optimizer step).
2. Score next chunk of eval data with the updated LoRA.

Adding phase 4 means: 33% more TTT compute, 33% more LoRA adaptation,
33% more eval data scored under increasingly-adapted weights.

Memory `project_eval_time_quant_repair_idea` confirms ~100-180s eval
headroom. A canonical eval at NL=2 takes ~423-495s; one extra phase
adds ~1-2 minutes (~70-120s). Should fit within the eval cap.

**This isolates the TTT-compute axis from the LOOP_PATTERN axis.**
If 091 wins where 080 / 083 don't, the eval-time-headroom lever is
better spent on TTT than on deeper recurrence. If 091 ≈ canonical,
adding phases doesn't help (TTT already converges in 3 phases).

## Baseline

060A canonical post-TTT post-quant val_bpb (PHASED_TTT_NUM_PHASES=3,
NL=2, the leaderboard number for seed 42).

## Expected Δ vs 060A canonical

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (extra phase yields more LoRA adaptation) | -0.0005 to -0.0015 | medium |
| Plausible (TTT already converged at 3 phases) | -0.0001 to +0.0005 | medium-high |
| Worst (4th phase overfits to prefix distribution) | +0.0005 to +0.0020 | low-medium |

Lower regression risk than LOOP_PATTERN-based specs because the
trained model isn't being asked to do anything OOD; just more
adaptation cycles of the existing TTT mechanism.

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0010

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = ""          # unset — use canonical contiguous-band logic
NUM_LOOPS    = 2           # canonical
LOOP_START   = 3           # canonical
LOOP_END     = 5           # canonical
ENABLE_LOOPING_AT = 0.0    # eval-only (immediately active)
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 1
PHASED_TTT_ENABLED = 3
PHASED_TTT_NUM_PHASES = 4   # ← THE LEVER (was 3)
PHASED_TTT_PREFIX_DOCS = 2500
TTT_LORA_RANK = 80
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

(All other env vars per 060A defaults.)

## Code changes

**None.** `PHASED_TTT_NUM_PHASES` already exists in 060A baseline
config, just set to a non-default value.

### Compile-graph audit

Eval-only run with 4 TTT phases instead of 3. `PHASED_TTT_NUM_PHASES`
is a Python integer used as a `range()` bound for the TTT loop. The
compiled `forward_ttt` graph is called once per phase. **The graph
itself is unchanged** — only the number of times it's called changes.

**Zero new graph variants. No mid-run recompile possible.**

The TTT LoRA Adam optimizer state evolves across phases (this is by
design). No graph implications — Adam state is stored as Python
optimizer attributes, not in the compiled graph.

`forward_logits` graph is also unchanged from canonical (LOOP_PATTERN
unset, contiguous-band logic with `num_loops=2, loop_start=3,
loop_end=5`).

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~30-32 min total. Adds ~1-2 min vs canonical 25-30 min:
  - Same compile burst (no new graph variants)
  - Same pre-quant eval at canonical pace
  - Same GPTQ
  - Post-quant eval + TTT 4 phases (vs 3) ≈ +1-2 min

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/091-eval-more-TTT-phases/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log; should show "TTT phase 4/4" at end
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (canonical isn't broken; this is sanity)
- Post-TTT val_bpb > 1.075 → kill (TTT 4 phases catastrophically worse)
- Total wallclock > 35 min → kill (time budget exceeded)

## Cost estimate

~$3-4 (full pipeline eval, ~30-32 min on 4×H100).

## Sequencing

This spec is **independent of 080-090**. It tests a different lever:
TTT compute axis at canonical recurrence. Can run in any order
relative to the others.

## Reading 091 alongside 080-090

| Lever | Spec | What it tests |
|---|---|---|
| Recurrence depth | 080/081/082 | layer-pass count |
| Recurrence depth + TTT | 083/084 | layer-pass count under TTT |
| Recurrence shape | 085-088 | matched-compute pattern variants |
| Recurrence position | 089/090 | band position |
| **TTT compute** | **091** | **phase count** |

091 is the cleanest "single-lever" test of "is the eval-time-headroom
better spent on TTT or on recurrence?"

## Followup specs gated on result

- If 091 wins: try PHASED_TTT_NUM_PHASES=5 (spec 092 candidate).
- If 091 wins AND any 083/084 wins: combine — TTT phases up + deeper
  recurrence (spec 093 candidate).
- If 091 ≈ canonical: TTT already saturates at 3 phases; eval headroom
  better spent elsewhere (e.g., recurrence).
- If 091 loses: 4 phases overfit; sticking at 3 phases is correct.

## See also

- `research/specs/060L-ttt-prefix-up.md` (if exists) — sibling lever
  (longer prefix instead of more phases)
- `research/specs/060J-ttt-phases-up.md` (if exists) — may already
  cover this; verify before launching 091
- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — TTT-on
  template
- `research/ideas/parallelize-deep-looks.md` cluster SS (W12)
