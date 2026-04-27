# Spec 050D — Inject-before-activation up-LoRA on 1797 baseline

**Date:** 2026-04-28
**Branch:** `exp/050D-lora-untied` @ `c1d97f9`
**Idea:** follows `research/ideas/lora-on-1797.md` (050B series)

## Hypothesis

050B proved LoRA delta gets zero gradient forever: `leaky_relu²'(0) = 0` when `delta = A@B = 0` at init. Fix: inject `delta_up` BEFORE the MLP activation so the gradient evaluates at `W_up @ x` (nonzero from step 0). Remove the down LoRA entirely (half the params, simpler). Each loop pass gets a distinct up-projection, genuinely untying the MLP across passes.

## Baseline

Spec 050A (`f5b8af8` baseline, pre-quant EMA ~1.067).

## Expected Δ

−0.001 to −0.003 pre-quant vs 050 baseline. Higher confidence than 050B since B can now learn.

## Accept criteria

- Pre-quant EMA val_bpb ≤ 1.067.
- `lora_norms` log shows B_up growing from 0 (non-zero by step 500).
- No compile hang at loop activation.

## Config diff (vs spec 050)

```
LOOP_SCALE_INIT=recip
LOOP_ITER_EMBEDS=1
LOOP_FFN_LORA_RANK=2
```

Same env vars as 050B — no new knobs.

## Code changes

Branch: `exp/050D-lora-untied` @ `c1d97f9`
Train script: `records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py`

Key changes vs `exp/050B-lora-on-1797`:

```python
# Block.forward — inlined MLP, delta injected before activation:
if lora_delta_up is not None and not getattr(self.mlp, "_calib", False):
    h_pre = F.linear(x_flat, up_w) + F.linear(x_flat, lora_delta_up)
    h_act = self.mlp._activate(h_pre)     # gradient at W_up@x, not at 0
    mlp_out = F.linear(h_act, down_w)
else:
    mlp_out = self.mlp(x_normed, up_w, down_w)  # _calib fallback fires GPTQ hooks

# _compute_lora_deltas — returns delta_up tensor [P,L,hidden,dim], not tuple
# (down LoRA params removed entirely)

# enc/dec loops — gate on looping_active for zero pre-loop overhead:
if self._enc_ffn_lora_info is not None and self.looping_active:
    ...
```

Compile safety: `looping_active` gate → pre-loop blocks all pass `None` → single graph variant. Startup `_run_cu_bucket_warmup` already does backward with `looping_active=True`, so Graph_B_grad is pre-compiled at startup.

## Hardware ladder

- **4×H100 mini** — 20-min screen. Required (new code path, need smoke).
- **8×H100 official** — if mini shows ≥ −0.001 improvement.

## Seed plan

Single seed (42) for mini.

## Inputs

`/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...` No hotstart.

## Checkpoints

`final_model.pt` from GPTQ.

## Stop-early criteria

NaN, train_loss > 7.0 after step 200, step time > 1.5× at loop activation.
If `lora_norms` log shows B_up still 0.00000 at step 500 → halt, gradient barrier still present.

## Cost estimate

~$3 (20-min screen, 4×H100).

## Open questions for interview

1. Confirm `lora_norms: B_up=` grows from 0 by step 500.
2. Watch step time at loop activation — should be clean compile hit (no hang).
3. Pre-loop throughput should match clean 050A baseline (LoRA gated off pre-loop).
