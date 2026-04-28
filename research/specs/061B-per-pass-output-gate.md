# Spec 061B — Per-step output gate on encoder + decoder loop

**Status:** FROZEN — code committed at `926ab35` on `exp/061B-per-step-gate`,
pushed to fork. Ready to run.

**Date:** 2026-04-29
**Branch:** `exp/061B-per-step-gate`
**Pinned commit:** `926ab35d3ee01d3d5e84151b32272692495aa936`
**Parent:** 060A (#1855 port; forked from `exp/060-resume-ckpt @ a0a48b7`).

## Hypothesis

Two literature findings converge on this intervention:

1. **"Loop, Think & Generalize" (arXiv 2604.07822)** reports
   *"the margin increases with inference iterations until reaching a peak,
   then consistently declines"* — overthinking degradation in
   recurrent-depth transformers.
2. **"Two-Scale Latent Dynamics" (arXiv 2509.23314)** finds *"loop-step
   sizes shrink rapidly... mean ‖Δ(k)‖₂ decays quickly with k"* and
   *"consecutive updates become more orthogonal."*

A learnable scalar per step in the expanded encoder/decoder index list
(one gate per visit of each layer when looping is active) lets the
optimizer down- or up-weight any step. If late visits really do
contribute diminishing or noisy returns, the gate values bend below
1.0; if they contribute orthogonal value, gates stay near 1.0 and we
lose nothing. ~14 scalars total for default Loop345 expansion.

This is the cheapest possible test of structural per-step
differentiation — prior per-pass attempts (047C/D, 050B/D, 053, 054)
all spent thousands to millions of parameters. None won. This spec
strips the question to its floor: does *any* per-step scaling help, at
near-zero param cost?

**Note on framing:** the architecture's loop is implemented as the same
3 layers (3,4,5) repeated in the expanded `encoder_indices` /
`decoder_indices` lists, so "per-pass" in the original idea maps to
"per-step in expanded list." Same intervention; correct terminology.

## Baseline

060A.

## Expected Δ

**−0.0003 to −0.002 BPB.** Lower confidence; small mechanism with
theoretical support but no direct empirical precedent on this stack.
Likely-zero floor: gates may converge to ~1.0 and behave like baseline,
in which case the hypothesis is falsified cheaply.

## Accept criteria

- post-quant + post-TTT val_bpb ≤ 060A − 0.0003 (1-seed screen)
- no NaN, no compile pathology
- final per-step gate values logged from the saved checkpoint (sanity
  check: did they move from init? if all gates within ±0.02 of 1.0,
  that's the null result)

## Config diff vs 060A

```
PER_STEP_GATE_ENABLED = 1
```

(All other env vars unchanged from 060A.)

## Code changes

Three sites in `train_gpt.py`:

1. New env var `PER_STEP_GATE_ENABLED` (default 0).
2. `GPT.__init__` adds `self.per_step_gates_enc` and
   `self.per_step_gates_dec` as `nn.Parameter(torch.ones(...))` sized
   to the expanded encoder/decoder index lists when enabled; `None`
   when disabled.
3. `_forward_hidden` and `forward_ttt` both apply
   `x = x_prev + gate * (x - x_prev)` after each block call when
   `per_step_gate_enabled and looping_active`. Identity at gate=1.0.

The conditional is gated by an integer Python flag
(`gate_active = self.per_step_gate_enabled and self.looping_active`)
which is stable per-graph; combined with the existing
`looping_active` toggle this introduces at most one new compile graph
variant.

Diff stat: `1 file changed, 48 insertions(+), 2 deletions(-)`.

## Hardware ladder

- **Mini rung: 2×H100, 5-min smoke** — required (code change). Verify:
  - gates appear in optimizer's param list (use scalar_lr group)
  - no compile recompile mid-run
  - gradient flows (gate values move off init even slightly)
- **Official rung: 8×H100, 1 seed (42)**, 600s train + 600s eval.

If mini smoke shows compile pathology, halt and fix.

## Seed plan

1 seed (42) for screening.

## Inputs

Standard 060A paths; no hotstart; fresh training.

## Checkpoints emitted

- `final_model.pt` — pre-quant post-EMA
- `final_model.int6.ptz` — post-quant submission
- `train.log` — full log including phased TTT eval line

Saved to `/workspace/runs/061B-per-step-gate/seed_42/`.

## Stop-early criteria

- train_loss > 5.0 at step 1000 → kill
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- mid-run torch.compile recompile → kill (always-tensor violated)

## Cost estimate

~$1 mini + ~$5 official = **~$6**.

## Open questions for interview

1. **Gate optimizer group.** Gates are scalars; should land in
   `scalar_lr` Adam group (they're not matrix params). Confirm in
   smoke.
2. **What halts** if gates all move within ±0.02 of 1.0 at end? Treat
   as null; family killed; do not run multi-seed.
3. **Logging the final gates.** Execution should grep
   `per_step_gates_enc` / `_dec` from the saved `final_model.pt` and
   include the values in the run notes.

## Followups (NOT in this spec)

- 061B-fine: split into separate gates for attn vs MLP contribution
  if gates move and signal is positive but small.
- 061B-init: anti-overthinking init schedule (init last gate at 0.5)
  if gates land below 1.0 by training end.
- 063-combined: stack with 061A skip-layer-attn if both individually
  clean (061A still pending implementation).
