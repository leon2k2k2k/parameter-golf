# Spec 046E — Post-quant LayerNorm/bias fitting loop (Tier 2)

**Slug:** `046E-postquant-lnfit`
**Created:** 2026-04-27
**Status:** READY (code landed in commit 381baf2)
**Branch:** `exp/046-quant-repair`
**Commit:** `381baf2`
**Parent:** spec 046, idea Q3 in `research/ideas/quant-repair.md`

## Hypothesis

After GPTQ quantizes weights, the quantization error shifts layer-output
distributions. LayerNorm scale/shift parameters (and biases, if present) were
trained against pre-quant activations and now mismatch the post-quant ones.

Re-fitting LN scales/shifts on a small calibration set (matching pre-quant
fp32 activations) can recover that distribution shift at zero artifact-byte cost
(LN params already exist).

Designed in PR #1818-B (taka6745, never implemented). Designer's projection:
**−0.005 to −0.020 BPB**. Even at the low end, this would be the largest
quant repair effect since SDClip.

## Why this is the most interesting Tier 2 candidate

- Designer's projected gain (−0.005 to −0.020) dwarfs anything else in the queue
- Mechanism is principled and straightforward (block-wise re-fit against fp32 ref)
- ~30 lines of code
- Zero artifact-byte cost (LN params already in state_dict)
- Works in our existing RESUME_FROM_CKPT pipeline (post-GPTQ operation)
- **Never been tried by anyone in the comp**

## Code (landed in 381baf2)

`fit_passthrough_params_to_match_base()` in train_gpt.py:
- Iterates eval_model.named_parameters(); freezes int6 matrix weights;
  fits **all small (numel <=65536) fp16-passthrough params jointly**
- Per-block params fitted: `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`
- Global params fitted: `recur_alpha/beta`, `smear_lambda`, `parallel_*_lambdas`,
  `skip_gates/weights` if present
- Objective: per-batch MSE between eval_model.forward_logits(x) (post-quant, bf16)
  and base_model.forward_logits(x) (pre-quant, bf16), accumulated to fp32
- Optimizer: AdamW(lr=POSTQUANT_LNFIT_LR, betas=(0.9,0.95), wd=0)
- Calibration source: ShuffledSequenceLoader on train data (matches GPTQ source)

**Important caveat — in-memory only**: fitted values live on `eval_model` until
quantized eval finishes. The serialized artifact (`final_model.int6.ptz`) keeps
the pre-fit values. If a winning arm emerges, productionization requires
re-serialization (separate work).

Note that since RMSNorm in this codebase has no learnable params (line 848-854),
"LN fit" is a misnomer — we're actually fitting the per-block scaling parameters
(attn_scale/mlp_scale/resid_mix) which serve a similar functional role.

Env vars (default OFF):
- `POSTQUANT_LNFIT_ENABLED=1` to activate
- `POSTQUANT_LNFIT_ITERS` (default 5) — outer epoch count
- `POSTQUANT_LNFIT_LR` (default 1e-3) — AdamW learning rate
- `POSTQUANT_LNFIT_BATCHES` (default 8) — inner loop batch count per epoch

## Arms (after code lands)

| Arm | n_iters | calib batches | Notes |
|---|---|---|---|
| 046E-iters5 | 5 | 8 | designer's spec |
| 046E-iters10 | 10 | 8 | extended fit |
| 046E-iters5-cb16 | 5 | 16 | more calib data |
| 046E-iters5-mlp-only | 5 | 8 | only MLP-block LN (test localization) |

## Predicted outcome (designer's projection)

- Optimistic: 1.0556 (−0.0190 from baseline) — strong-strong win
- Realistic: 1.0710 (−0.0035) — clear win
- Pessimistic: 1.0746 (~null) — designer's projection wrong

## Launch form (per arm)

Same as 046 verification spec, plus the fit env vars:

```bash
# 046E-iters5 (designer's spec — start here)
export POSTQUANT_LNFIT_ENABLED=1
export POSTQUANT_LNFIT_ITERS=5
export POSTQUANT_LNFIT_LR=1e-3
export POSTQUANT_LNFIT_BATCHES=8
export RUN_ID="046E-postquant-lnfit-iters5"

# 046E-iters10 (extended fit)
export POSTQUANT_LNFIT_ITERS=10
export RUN_ID="046E-postquant-lnfit-iters10"

# 046E-iters5-cb16 (more calib data per epoch)
export POSTQUANT_LNFIT_ITERS=5
export POSTQUANT_LNFIT_BATCHES=16
export RUN_ID="046E-postquant-lnfit-cb16"

# 046E-iters5-low-lr (more conservative)
export POSTQUANT_LNFIT_LR=3e-4
export RUN_ID="046E-postquant-lnfit-lowlr"
```

Plus standard armD env vars (LOOP_ITER_EMBEDS=0 MLP_ONLY_FROM_PASS=0 LOOP_SCALE_INIT=recip)
and RESUME_FROM_CKPT path.

## Acceptance

Reference = 046 verification quantized (1.07467).

- **Strong-strong win**: < 1.060 (designer's optimistic projection — would be a major lever)
- **Strong win**: < 1.070 (clear -0.005)
- **Win**: < 1.0735 (clear improvement)
- **Noise**: 1.0735–1.0760 (no benefit; in-memory fit not enough OR optimizer divergent)
- **Kill**: > 1.0760 (fit hurts; LR too high or wrong objective)

**Watch for fit divergence**: if `avg_mse` increases between iterations, kill that arm.

## Cost

~$1-2 per arm × 4 arms = ~$5-8 total. Each arm runs ~5-7 min for the eval +
~1-2 min for the fit (5-10 iters × 8-16 batches × 2 forward passes).

**This is the highest-EV experiment in the entire 046 family.** Worth doing
once 046 verification + 046A-D config sweeps are done.
