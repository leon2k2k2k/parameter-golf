# Spec 086 — Eval-time matched-compute broad pattern WITH TTT on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W7)
**Branch:** `exp/071-loop-pattern` (same as 080-085)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: `parallelize-deep-looks.md`
cluster AA1 (W6) extended with TTT.

## Hypothesis

Sibling to 085 (matched-compute broad-pattern eval, TTT off). Tests
whether the broad-pattern pre-quant signal (if any) survives TTT and
GPTQ — i.e., whether it would translate to a real leaderboard
submission.

Pattern body: `1,2,3,3,4,5,4,5,6,5,6,7,7` — same as 085, 17 total
layer-passes (matched canonical compute), broader band {3..7} with
layers 6,7 visited 2× each.

Composes with the 080→083 / 081→084 lineage:

| Spec | Compute | Shape | TTT | What it adds |
|---|---|---|---|---|
| 080 | 20 (NL=3 eq) | canonical+ | off | pre-quant deeper-NL test |
| 083 | 20 (NL=3 eq) | canonical+ | on | leaderboard-relevant deeper-NL test |
| 085 | 17 (matched) | broad {3..7} | off | pre-quant shape test |
| **086** | **17 (matched)** | **broad {3..7}** | **on** | **leaderboard-relevant shape test** |

Reading 085+086 together:
- If 086 < canonical post-TTT and 085 ≈ baseline: TTT amplifies the
  shape effect.
- If both win: shape lever is real and survives the full pipeline;
  promote.
- If both fail: shape test closes; canonical band is genuinely tuned
  for both pre-quant and post-TTT outcomes.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical post-TTT

Tighter prediction than usual since weights are seed-matched:

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (broad pattern + TTT compose) | -0.0005 to -0.0015 | low |
| Plausible (TTT compensates / neutralizes) | -0.0003 to +0.0005 | medium |
| Likely (pattern shape OOD; TTT can't fix) | +0.0005 to +0.0020 | medium-high |
| Worst (TTT destabilized) | +0.0020 to +0.0050 | low |

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Win-conditional-on-085:** if 085 wins, 086 should win by similar
  or more. If 085 loses, 086 expected to lose at least as much.
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0010

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = "1,2,3,3,4,5,4,5,6,5,6,7,7"   # same as 085
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

Index lists (verified earlier in 074/085 design):
- `encoder_indices = [0, 1, 2, 3, 3, 4, 5, 4]` (8 indices)
- `decoder_indices = [5, 6, 5, 6, 7, 7, 8, 9, 10]` (9 indices)
- 17 total layer-passes.

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080-085.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. Two distinct compiled
functions (forward_logits, forward_ttt) each compile once on first
respective invocation with the new index lists. Both produce one
graph variant each. No mid-run recompile.

The TTT path uses `_block_with_lora` wrapping each Block call. LoRA
rank fixed at 80 (canonical 060A setting). LoRA shapes unchanged.

Bank weights load from 060A checkpoint. Layer-indexed; visits to
layers 6,7 (twice each in this pattern) re-read the loaded layer-6
and layer-7 banks without modification. **No weight slicing in
compile.** **No narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min. Breakdown:
  - Compile burst 1 (forward_logits): ~3-4 min
  - Compile burst 2 (forward_ttt): ~3-4 min
  - Pre-quant eval at the broad pattern: ~5-6 min (matched compute,
    similar to canonical pace)
  - GPTQ calibration + quantization: ~3-5 min
  - Post-quant eval + TTT 3 phases: ~5-7 min

Total: 25-30 min. Cost: ~$3-4.

## Seed plan

1 seed (42) — matches 060A.

## Inputs / Outputs

- Output dir: `/workspace/runs/086-eval-broad-pattern-TTT/seed_42/`.

Key artifacts:
- `train.log` — full pipeline log
- `final_model.int6.ptz` — submittable artifact at the broad pattern
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.075 → kill
- Compile time on either compiled function > 6 min → kill
- NaN → kill
- Total wallclock > 35 min → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

**Recommended:** run 085 first. If 085 lands within noise of
canonical or better, 086 worth running. If 085 catastrophically
loses (kill range), abort 086 — TTT won't fix a fundamentally
broken pattern.

## Reading 085 + 086 together

The matched-compute pattern-shape question is independent of the
deeper-recurrence (compute axis) question:

- 085 + 086: **shape** axis at fixed compute
- 080/083 / 081/084: **compute** axis at canonical shape
- 060A: pivot point at compute=17, canonical shape

If 086 wins: write a multi-shape sweep spec to find the optimal
non-canonical pattern. Worth promoting at multi-seed.

## Followup specs gated on result

- If 086 wins: 087 candidate = TTT-on stepped pattern (073-shape)
  for shape-sweep continuation.
- If 086 ≈ canonical: pattern shape lever doesn't exist post-TTT.
- If 086 loses: pattern-shape-at-eval thread closes.

## See also

- `research/specs/085-eval-time-broad-pattern-on-060A.md` — TTT-off sibling
- `research/specs/074-loop-pattern-broad-17pass.md` — training-time variant
- `research/specs/083-eval-time-NL3eq-with-TTT-on-060A.md` — TTT-on
  template for full-pipeline eval
- `research/ideas/parallelize-deep-looks.md` cluster AA (W6), W7 sequencing
