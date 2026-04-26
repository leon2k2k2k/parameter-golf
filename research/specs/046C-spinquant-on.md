# Spec 046C — SpinQuant ON ablation

**Slug:** `046C-spinquant-on`
**Created:** 2026-04-27
**Status:** READY (after 046 PASS)
**Branch:** `exp/046-quant-repair`
**Commit:** `0ea6a97`
**Parent:** spec 046 (verification), idea Q2 in `research/ideas/quant-repair.md`

## Hypothesis

SpinQuant code is wired in baseline (`spinquant_hotstart.py`, env vars set up)
but **disabled by default** (`SPINQUANT_ENABLED=0`). Four prior PR ablations
agree rotation methods are essentially null on Muon-trained weights:
- #670 (abaybektursun): −0.0002 BPB (basically zero)
- #1224 (vermissa0ss): −0.0000567 BPB (zero)
- #1718 (himanshudongre): "null on Muon sub-Gaussian weights"
- #1695 (X-Abhishek-X): SpinQuant V1 ported, quant tax unchanged at +0.012

Mechanism cited: Muon orthogonalization → sub-Gaussian weights; GPTQ act-order
+ Cholesky already handles outliers; rotation has nothing left to do.

**However**, all four ablations predate LQER + CaseOps + current Muon-WD ratios.
Whether the conclusion holds on the post-LQER/CaseOps stack is unverified.

This run **settles the question** in our exact stack with one $1 experiment.

## Predicted outcome

±0.001 BPB. Most likely null per prior evidence. If SpinQuant + LQER interact
in some unexpected way (e.g., rotation changes which tensors LQER picks for
top-K residual), we could see a small effect either direction.

## Arms

| Arm | Change | Notes |
|---|---|---|
| 046C-spinquant-on | `SPINQUANT_ENABLED=1`, sites = baseline default | Single arm; the question is binary |

Optionally: 046C-spinquant-mlp-only (`SPINQUANT_SITES='mlp_in,mlp_proj_in'`)
if 046C-spinquant-on shows a directional effect — to localize.

## Config diff (vs 046 verification baseline)

```bash
SPINQUANT_ENABLED=1
SPINQUANT_SEED=42
SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'
RUN_ID="046C-spinquant-on"
```

All other env vars identical to 046 verification.

## Acceptance

Reference = 046 verification quantized (~1.07467).

- **Surprise win**: < 1.0735 (rotation has new effect with LQER) → run mlp-only arm to localize
- **Noise/null**: 1.0735–1.0760 (matches prior PR evidence) → close direction definitively
- **Hurts**: > 1.0760 (negative interaction with LQER) → important finding

Pre-quant should be **identical** to 046 (rotation only affects quant).
If pre-quant differs, the rotation infra is touching weights at load time
unexpectedly — investigate.

## What to watch

- `SpinQuant:` log lines at startup (verifies infra activated)
- Pre-quant val_bpb (should equal 046 verification's pre-quant)
- Quantized val_bpb (the headline)
- Submission size (rotation typically makes weights compress slightly worse)

## Decision

| Result | Next |
|---|---|
| Win | Adopt; combine with 046A/B winners |
| Noise | Close SpinQuant direction permanently; update memory |
| Hurts | Important finding — prior nulls were on different stacks; LQER changes things |

## Cost

~$1 single arm. ~$2 if mlp-only follow-up runs.
