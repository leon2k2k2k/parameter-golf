# Spec 097 — Eval-time TTT_BATCH_SIZE=32 (smaller batch) at canonical NL on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, full TTT+GPTQ pipeline.

**Date:** 2026-04-29 (autonomous wake W18)
**Branch:** `exp/071-loop-pattern` (same as 080-096)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster KKK1 (W18).

## Hypothesis

`TTT_BATCH_SIZE=64` is the default. The TTT optimizer (Adam over
LoRA) accumulates gradients over batches of this many sequences
before applying an update. Smaller batches → more Adam updates per
TTT phase → finer-grained adaptation.

This spec sets `TTT_BATCH_SIZE=32` (half the default). At the same
total TTT compute (3 phases × 2500 prefix docs), this means roughly
**2× as many Adam updates** vs canonical, each on noisier gradients.

**Hypothesis evolution:**
- (a) **More frequent updates help:** the LoRA traverses the loss
  landscape with finer granularity; per-doc adaptation improves.
  096 wins.
- (b) **Larger batches were better-tuned:** smaller batches add
  gradient noise without compensating signal benefit. 097 ≈ canonical
  or hurts.
- (c) **Match-LR-to-batch wisdom:** smaller batches could benefit
  from a larger LR; this spec doesn't tune LR, so may underperform.

This is a **novel TTT axis** distinct from phase count (091),
LoRA LR (094/095), and momentum (096). Together these specs sample
the TTT hyperparameter space.

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number
for seed 42).

## Expected Δ vs 060A canonical

| Outcome | Δ post-TTT val_bpb | Likelihood |
|---|---|---|
| Best (smaller batches help) | -0.0005 to -0.0015 | low-medium |
| Plausible (insensitive to batch size) | -0.0001 to +0.0005 | medium-high |
| Likely (canonical batch was tuned) | +0.0005 to +0.0020 | medium |
| Worst (gradient noise dominates) | +0.0020 to +0.0040 | low-medium |

## Accept criteria (post-TTT, post-quant)

- **Win:** post-TTT val_bpb ≤ **(060A canonical post-TTT) − 0.0005**
- **Noise:** within ±0.0005 of canonical
- **Kill:** post-TTT val_bpb ≥ canonical + 0.0020

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = ""
NUM_LOOPS    = 2
LOOP_START   = 3
LOOP_END     = 5
ENABLE_LOOPING_AT = 0.0
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 1
PHASED_TTT_ENABLED = 3
PHASED_TTT_NUM_PHASES = 3
PHASED_TTT_PREFIX_DOCS = 2500
TTT_LORA_RANK = 80
TTT_LORA_LR = 1e-4
TTT_BETA1 = 0.0           # canonical
TTT_BETA2 = 0.99          # canonical
TTT_BATCH_SIZE = 32       # ← THE LEVER (canonical 64)
TTT_CHUNK_SIZE = 48       # canonical (unchanged)
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

## Code changes

**None.** `TTT_BATCH_SIZE` is an existing env-var-readable parameter.

### Compile-graph audit

Eval-only run, full TTT+GPTQ pipeline. `TTT_BATCH_SIZE` controls how
many sequences are accumulated per Adam step in the TTT loop. This
is **a Python-level data-loop parameter**, not a graph parameter.

The compiled `forward_ttt` graph operates on a single sequence's
forward at a time (or a fixed micro-batch); changing how many such
calls happen between Adam steps doesn't change the compiled graph.

**Zero new graph variants from this lever.** **No mid-run recompile
possible.**

The compiled forward_logits graph is identical to 060A canonical (no
LOOP_PATTERN). All graphs pre-compiled during loop_warmup.

LoRA shapes determined at __init__ by `TTT_LORA_RANK=80`, unchanged.

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only (full pipeline).**
- Wall: ~25-30 min. Possibly +2-3 min from the additional Adam updates
  (each adds small Python-side overhead, but the GPU forward dominates).

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/097-eval-TTT-batch-32/seed_42/`.

Key artifacts:
- `train.log`
- `final_model.int6.ptz` — submittable artifact
- `final.json` — pre-quant + post-TTT val_bpb numbers

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-TTT val_bpb > 1.075 → kill
- NaN → kill

## Cost estimate

~$3-4 (full pipeline eval, ~25-30 min on 4×H100).

## Sequencing

This spec is independent of 080-096. Tests TTT-batch-granularity
axis. Test in any order.

## Reading 091/094/095/096/097 together (TTT hyperparameter map)

| Lever | Spec | Direction |
|---|---|---|
| Phase count | 091 | up |
| LoRA LR | 094/095 | down/up |
| Momentum (β1) | 096 | up (0→0.9) |
| **Batch size** | **097** | **down (64→32)** |

If 097 wins: more-frequent-updates is a real lever; canonical
batch size was suboptimal.
If 097 ≈ canonical: batch size insensitive at this regime.
If 097 loses: canonical batch was right-tuned.

Reading the full TTT hyperparameter map gives a 5D sample of which
TTT directions matter. Each axis is independent and can be combined
post-hoc if multiple win.

## Followup specs gated on result

- If 097 wins: try TTT_BATCH_SIZE=16 (push further), spec 098
  candidate.
- If 097 ≈ canonical: TTT-batch-axis closes; pivot to chunk size (LLL).
- If 097 loses: confirm canonical batch size; close downward direction.

## See also

- `research/specs/091/094/095/096` — sibling TTT-axis specs
- `research/ideas/parallelize-deep-looks.md` cluster KKK (W18)
