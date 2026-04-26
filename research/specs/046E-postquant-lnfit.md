# Spec 046E — Post-quant LayerNorm/bias fitting loop (Tier 2)

**Slug:** `046E-postquant-lnfit`
**Created:** 2026-04-27
**Status:** DRAFT — needs ~30 lines of code first
**Branch:** `exp/046-quant-repair` (will need new commit)
**Commit:** TBD (after code lands)
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

## Code change required (separate spec/commit)

Add post-quant fitting loop in `serialize()` or a new function called between
GPTQ and final artifact write:

```python
def fit_layernorm_post_quant(base_model, quantized_state, calib_loader, n_iters=5):
    """
    Re-fit LN scale/shift to match fp32 (pre-quant) layer outputs.
    """
    # 1. Run base_model (fp32, pre-quant) on calib batches → save per-layer outputs
    # 2. Load quantized weights into a copy, freeze everything except LN scales/shifts
    # 3. AdamW loop (n_iters) minimizing MSE per-block-output vs saved fp32 outputs
    # 4. Return updated LN state_dict
```

New env vars:
- `POSTQUANT_LNFIT_ENABLED` (0/1, default 0)
- `POSTQUANT_LNFIT_ITERS` (default 5)
- `POSTQUANT_LNFIT_LR` (default 1e-3)
- `POSTQUANT_LNFIT_BATCHES` (default 8 calib batches)

Open questions for code design:
- Per-block (loop body) or per-layer-norm (atomic) optimization?
- Capture fp32 activations once at start, or re-compute each iter?
- Include biases in fit if linears have them? (verify CaseOps stack first)
- Fit per loop-iter pass separately, or share LN across passes?

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

## Pre-flight (before specing arms in detail)

- Verify CaseOps stack has biases on linears (relevant for fit scope)
- Inspect `Block` class in train_gpt.py to identify all LN params
- Decide per-block vs per-LN optimization unit
- Estimate code change complexity (target <30 lines)

## Cost

~30 min code + ~$1-2 per arm × 4 arms = ~$5-8 total.

**This is the highest-EV experiment in the entire 046 family.** Worth doing
once 046 verification + 046A-D config sweeps are done.
