# Spec 099 — Eval-time TTT-disabled (calibration diagnostic) on 060A

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint, GPTQ on, TTT disabled.

**Date:** 2026-04-29 (autonomous wake W20)
**Branch:** `exp/071-loop-pattern` (same as 080-098)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: cluster QQQ1 (W20).

## Hypothesis

**Calibration diagnostic spec.** At eval, run the full pre-quant +
GPTQ pipeline but **skip TTT entirely** (`TTT_ENABLED=0`).

Tells us: how much val_bpb does TTT contribute on top of pre-quant +
GPTQ? Sizes the TTT lever for interpretation of all TTT-axis specs
(091, 094, 095, 096, 097, 098).

If TTT contributes ~0.005 bpb total: any 0.001 TTT-axis improvement
is a 20% relative gain.
If TTT contributes ~0.001 bpb total: TTT is mostly noise; the
TTT-axis sweep specs are small contributions.

Without this calibration, the TTT-axis specs sit in absolute deltas
without context. **099 anchors the TTT scale.**

## Baseline

060A canonical post-TTT post-quant val_bpb (the leaderboard number).

## Expected Δ vs 060A canonical

The model was trained with TTT in mind; disabling it at eval should
hurt:

| Outcome | Δ post-quant val_bpb | Likelihood |
|---|---|---|
| Best (TTT contributes very little) | +0.0001 to +0.0010 | low |
| Plausible (TTT modest contribution) | +0.0010 to +0.0030 | medium |
| Likely (TTT significant contribution) | +0.0030 to +0.0080 | high |
| Worst (TTT essential, model nearly broken without it) | +0.0080 to +0.0150 | low-medium |

Memory `project_baseline_1851_post_1872` notes that "PR #1797
measurements" use the full TTT pipeline; without TTT, expect
substantial regression.

## Accept criteria

This spec is **diagnostic, not for promotion**. It calibrates the
TTT lever. Outcomes:

- **Δ < +0.0005:** suspicious — TTT may not be running correctly in
  the canonical 060A measurement (re-verify).
- **Δ +0.0005 to +0.0030:** TTT modest; TTT-axis specs (091,
  094-098) range over a small lever.
- **Δ +0.0030 to +0.0100:** TTT substantial; TTT-axis specs
  range over a meaningful lever.
- **Δ > +0.0100:** TTT critical; the model architecture relies on
  TTT for competitive bpb.

No "win" or "kill" thresholds — this is pure measurement.

## Config diff vs 060A canonical eval

```
LOOP_PATTERN = ""
NUM_LOOPS    = 2
LOOP_START   = 3
LOOP_END     = 5
ENABLE_LOOPING_AT = 0.0
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
TTT_ENABLED  = 0           # ← THE LEVER (canonical 1)
PHASED_TTT_ENABLED = 0     # ← AND THIS (canonical 3)
PHASED_TTT_NUM_PHASES = 0  # ← AND THIS (canonical 3)
GPTQ_CALIBRATION_BATCHES = 16
GPTQ_RESERVE_SECONDS = 4
```

(Other TTT params are now irrelevant: TTT_LORA_LR, TTT_BETA1,
TTT_BETA2, TTT_BATCH_SIZE, etc.)

## Code changes

**None.** `TTT_ENABLED` and `PHASED_TTT_ENABLED` are existing
env-var-readable parameters in 060A's hyperparam dump.

### Compile-graph audit

Eval-only run, GPTQ pipeline only (no TTT phases).

When TTT is disabled:
- `forward_logits` runs as canonical (no LOOP_PATTERN; contiguous-
  band logic). Same compiled graph as 060A canonical. Cached.
- `forward_ttt` is **never called**. Its compiled graph (from
  loop_warmup) is harmless dead code; no graph variant change.
- The TTT phase loop has 0 iterations; no Adam steps; no LoRA
  updates.

**Zero new graph variants.** **No mid-run recompile possible.**

GPTQ runs as canonical (same calibration batches). The eval forward
post-GPTQ uses the quantized weights, no LoRA layered on top.

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.** GPTQ on, TTT off.
- Wall: ~15-18 min total. Faster than canonical (skips TTT phase
  time, which is typically 5-7 min):
  - Compile bursts (forward_logits only, since forward_ttt skipped):
    ~3-4 min
  - Pre-quant eval: ~5-6 min
  - GPTQ calibration + quantization: ~3-5 min
  - Post-quant eval: ~3-4 min (no TTT phases)

Cost: ~$2.

## Seed plan

1 seed (42).

## Inputs / Outputs

- Output dir: `/workspace/runs/099-eval-TTT-disabled/seed_42/`.

Key artifacts:
- `train.log`
- `final_model.int6.ptz` — submittable artifact (without TTT-eval)
- `final.json` — pre-quant + post-quant val_bpb numbers (no
  post-TTT row since TTT skipped)

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill
- Post-quant val_bpb > 1.085 → kill
- NaN → kill

## Cost estimate

~$2 (eval-only, ~15-18 min on 4×H100 — TTT skip saves time).

## Sequencing

This spec is independent of 080-098. Run as a calibration first
(or last); reading-of-result informs interpretation of TTT-axis
specs.

## Reading 099 alongside the TTT-axis sweep

If 099's Δ vs canonical is X:
- 091 (more phases) gain: divide by X for relative contribution
- 094/095 (LR sweep) gains: same
- 096 (β1 momentum): same
- 097 (batch size): same
- 098 (β2): same

A TTT-axis spec with 0.0005 absolute Δ on top of canonical post-TTT
is:
- A **10% relative improvement** if TTT contributes 0.005 (X=0.005)
- A **50% relative improvement** if TTT contributes 0.001 (X=0.001)
- Same absolute, very different relative sizes.

This calibration matters for prioritizing followup work.

## Followup specs gated on result

- If 099 contributes < 0.0005 vs canonical: TTT axis is essentially
  closed; no point in further TTT-axis sweep.
- If 099 contributes 0.0005-0.005: TTT-axis sweep specs are
  meaningful; cherry-pick winning TTT-axis improvements.
- If 099 contributes > 0.005: TTT is a major lever; the TTT-axis
  improvements are amplified accordingly.

## See also

- `research/specs/091/094/095/096/097/098` — TTT-axis sweep that
  this calibrates
- `research/ideas/parallelize-deep-looks.md` cluster QQQ (W20)
