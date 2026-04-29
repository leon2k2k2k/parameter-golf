# Spec 082 — Eval-time NL=1-equivalent (no recurrence) on 060A checkpoint

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on saved 060A checkpoint. No new code.

**Date:** 2026-04-29 (autonomous wake W3)
**Branch:** `exp/071-loop-pattern` (same as 080/081)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's `final_model.pt`. Source: `parallelize-deep-looks.md`
cluster M1 (W2).

## Hypothesis

Bookend to the 080 / 081 scaling curve from BELOW. The 060A model
trained at NL=2 (3 passes through {3,4,5}). 080 tested NL=3 eq
(4 passes), 081 tested NL=4 eq (5 passes). This spec tests **NL=1 eq
(1 pass each = no recurrence)** — the *minimum* iteration count.

Per spec 040 ("no-loop ablation," trained without looping): pre-quant
EMA val_bpb landed at **1.07222**, vs canonical 1.06358 — a gap of
**+0.00864**. But 040's *weights* were also trained without looping;
its model is fundamentally different from 060A's.

This spec is a stricter test: **same trained-with-looping weights, but
forward with each layer visited only once.** Expected to be worse than
060A canonical because the model was optimized for NL=2 dynamics, but
*how much worse* is informative:

- If 082 ≈ 040 (gap ~+0.0086): the recurrence-trained weights are not
  capable at NL=1; the loop is essential.
- If 082 < 040 (smaller gap): the trained weights have *some* NL=1
  capability — the loop adds refinement, not foundational predictive
  power.
- If 082 > 040: trained-with-loop weights are *worse* at NL=1 than
  trained-without-loop weights. Indicates the model has specialized
  hard for NL=2 dynamics.

Combined with 080/081, builds a 4-point scaling curve:
- NL=1 (082): pattern "1,2,3,4,5,6,7", 11 total passes
- NL=2 (canonical, 060A): 17 total passes — **baseline**
- NL=3 (080): 20 total passes
- NL=4 (081): 23 total passes

## Baseline

Same as 080/081: 060A pre-quant post-EMA val_bpb at canonical NL=2
(~1.06358 from `runs/060A-1855-port/seed_42/`).

## Expected Δ vs 060A

| Outcome | Δ bpb | Likelihood |
|---|---|---|
| Best (recurrence is mostly redundant) | +0.0010 to +0.0030 | very low |
| Plausible (recurrence helps but not load-bearing) | +0.0040 to +0.0080 | medium |
| Likely (matches 040 — recurrence is essential) | +0.0080 to +0.0120 | high |

This is expected to **lose**. The interest is the magnitude.

## Accept criteria

This spec is **diagnostic, not for promotion** — no submission outcome.
Treat it as a measurement that calibrates 080/081's gains:

- **Informative:** Δ vs 060A measured cleanly (any number tells us
  something).
- **Suspicious:** Δ < +0.0020 (indicates trained weights work at NL=1,
  contradicts 040). Re-verify.
- **Catastrophic kill:** pre-quant > 1.20 (model genuinely broken at
  NL=1). Investigate before drawing conclusions.

## Config diff vs 060A baseline

```
LOOP_PATTERN = "1,2,3,4,5,6,7"
NUM_LOOPS    = 2          # any positive value enables looping_active
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.0   # eval-only — active immediately
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
```

## Code changes

**None.** Same `LOOP_PATTERN` env var as 080/081. Pattern body is
shorter (7 elements), giving fewer total iterations.

### Compile-graph audit

Eval-only run. Model construction reads `LOOP_PATTERN` once at
`GPT.__init__`, building 11-element `encoder_indices=[0,1,2,3,4]` (5)
and `decoder_indices=[5,6,7,8,9,10]` (6). These lists are exactly
identical to the non-looping case (`looping_active=False`) which uses
`range(num_encoder_layers)` and `range(num_encoder_layers, num_layers)`.

So 082 functionally re-traces the *same* compiled forward graph as
the non-looping path — no new graph variant beyond what already exists
from training-time loop_warmup. **No mid-run recompile** because both
the looping and non-looping graphs are pre-compiled during the loop_warmup
phase (per `loop_warmup:enabled` line in train.log).

Bank weights load from 060A checkpoint via `load_state_dict`. Bank
indexing by layer index reads each layer's weights once. **No weight
slicing in compile.** **No narrow-K matmul.** **No dynamic shapes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.**
- Wall: ~5 min (NL=1 forward is *cheaper* than canonical — fewer
  layer-passes per batch).

## Seed plan

1 seed (42) — matches 060A.

## Inputs / Outputs

- Same as 080. Output dir: `/workspace/runs/082-eval-NL1eq/seed_42/`.

## Stop-early criteria

- Pre-quant val_bpb > 1.20 → kill (model genuinely broken)
- NaN in logits → kill

## Cost estimate

~$0.50 (eval-only, ~5 min on 4×H100; cheaper than 080/081 due to fewer passes).

## Open questions

1. **Validate 060A checkpoint location** (same dependency as 080/081).
2. **TTT/GPTQ off** for the cleanest pre-quant comparison.
3. **What to do if 082 lands much closer to canonical than expected?**
   That would be a surprising positive result: the trained recurrence
   isn't load-bearing. Promote a 4-point curve to a writeup.

## Reading 080 + 081 + 082 together

| Spec | NL eq | Total passes | Δ {3,4,5} visits/each | Expected bpb |
|---|---|---|---|---|
| **082** | **1** | **11** | **1** | **+0.005 to +0.012** |
| 060A | 2 | 17 | 3 | 1.06358 (baseline) |
| 080 | 3 | 20 | 4 | -0.001 to +0.001 |
| 081 | 4 | 23 | 5 | -0.002 to +0.002 |

Curve shape will tell us:
- If monotone-decreasing on [1, 4]: recurrence is contraction; deeper helps
- If U-shaped with minimum at 2: trained NL is the optimal point
- If L-shaped (cliff at NL<2, plateau at NL≥2): recurrence adds
  capacity up to a saturation point at NL=2

## See also

- `research/specs/080-eval-time-deeper-loop-on-060A.md` — sibling
- `research/specs/081-eval-time-NL4eq-on-060A.md` — sibling
- `runs/040-no-loop-ablation-screen/` — 040's no-loop training baseline (1.07222)
- `research/ideas/parallelize-deep-looks.md` cluster M (W2)
