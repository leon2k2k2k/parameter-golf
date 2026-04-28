# Spec 062 — Per-pass output gate on loop block

**Date:** 2026-04-29
**Branch:** `exp/062-per-pass-gate`
**Parent:** 060A (#1855 port on `research`).

## Hypothesis

Two literature findings converge on this intervention:

1. **"Loop, Think & Generalize" (arXiv 2604.07822)** reports that *"the margin
   increases with inference iterations until reaching a peak, then consistently
   declines"* — overthinking degradation in recurrent-depth transformers.
2. **"Two-Scale Latent Dynamics for Recurrent-Depth Transformers" (arXiv
   2509.23314)** finds *"loop-step sizes shrink rapidly... mean ‖Δ(k)‖₂ decays
   quickly with k"* and *"consecutive updates become more orthogonal."*

A learnable scalar per pass per loop layer (init 1.0) lets the optimizer
explicitly down- or up-weight any pass. If late passes really do contribute
diminishing or noisy returns, the gate values should bend below 1.0; if they
contribute orthogonal value, gates stay near 1.0 and we lose nothing.

15 trainable scalars total (3 loop layers × ~5 passes). Identity at init.

This is the cheapest possible test of structural per-pass differentiation —
prior per-pass attempts (047C/D, 050B/D, 053, 054) all spent thousands to
millions of parameters. None won. This spec strips the question to its
floor: does *any* per-pass scaling help, at near-zero param cost?

## Baseline

060A.

## Expected Δ

**−0.0003 to −0.002 BPB.** Lower confidence; small mechanism, theoretical
support but no direct empirical precedent on this stack. Likely-zero floor:
gates may converge to ~1.0 and behave like baseline (in which case the
hypothesis is falsified cheaply).

## Accept criteria

- post-quant + post-TTT val_bpb ≤ 060A − 0.0003 (1-seed screen)
- no NaN, no compile pathology
- final per-pass gate values logged to `pass_gates.json` (sanity: did they move
  from init? if all 15 stuck at 1.000, that's the null result)

## Config diff vs 060A

```
PER_PASS_GATE_ENABLED = 1
PER_PASS_GATE_INIT    = 1.0
```

## Code changes

- Add `self.pass_gates = nn.Parameter(torch.ones(NUM_LOOPS))` to each loop
  block (3 blocks × 5 passes ≈ 15 params total).
- In loop block forward, scale block contribution by
  `self.pass_gates[pass_idx]` *before* residual add:
  ```python
  out = block_fn(x)
  return x + self.pass_gates[pass_idx] * out
  ```
- Compile-safe: `pass_gates` is a static `nn.Parameter`, indexed by an integer
  that's already a graph constant per pass. No new graph variants.
- At end of training, dump gate values to `pass_gates.json` for analysis.

Branch `exp/062-per-pass-gate` from current `research`. Commit hash: **TBD** —
research must implement, push, and pin before mini rung.

## Hardware ladder

- **Mini rung: 2×H100, 5-min smoke.** Required (code change, even if tiny).
  Verify: gate values in optimizer's param list, gradient flows, no compile
  surprise.
- **Official rung: 8×H100, 1 seed (42)**, 600s train + 600s eval.

## Seed plan

1 seed (42).

## Inputs

Standard 060A paths; no hotstart.

## Checkpoints emitted

- `final_model.pt`, `final_model.int6.ptz`, `train.log`
- `pass_gates.json` — final values of all 15 scalars

Saved to `/workspace/runs/062-per-pass-gate/seed_42/`.

## Stop-early criteria

- train_loss > 5.0 at step 1000 → kill
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- mid-run recompile → kill (shouldn't happen; static parameter)

## Cost estimate

~$1 mini + ~$5 official = **~$6**.

## Open questions for interview

1. **Init value.** 1.0 (identity) is the safe default. Could try 0.5 (force
   compression) or per-pass-decreasing init (anti-overthinking prior). Default
   to 1.0; sweep only if it doesn't move.
2. **Apply to attn AND MLP separately, or the whole block?** Spec defaults to
   the whole block (one scalar per pass per layer). Finer-grained = 30 params
   instead of 15.
3. **What halts** if all 15 gates land within ±0.05 of 1.0 at training end?
   Treat as null result; family killed; do *not* run multi-seed.
4. **Compose with 061?** Independent. If both individually clean, freeze
   063-combined.

## Followups (NOT in this spec)

- 062b: split attn/MLP gates if 062 lands ambiguous.
- 062c: anti-overthinking init schedule (init pass-N−1 gate at 0.5).
- 063-combined: 061 + 062 if both individually clean.
