# Spec 046F — Short-burst AR self-generated calibration (Tier 2)

**Slug:** `046F-ar-selfgen-calib`
**Created:** 2026-04-27
**Status:** DRAFT — needs ~20 lines of code first
**Branch:** `exp/046-quant-repair` (will need new commit)
**Commit:** TBD (after code lands)
**Parent:** spec 046, idea Q4 in `research/ideas/quant-repair.md`

## Hypothesis

PR #756 ablation showed **AR self-generated calibration matches val-data calibration
within 0.0003 BPB** on this competition's stack — basically free signal. PR #1234
tried it but reserved 210s for AR generation, costing 30% of training steps and
netting +0.028 BPB worse. The lever is real, the budget was wrong.

In **eval-only mode** (RESUME_FROM_CKPT) there's no training-step cost. AR
generation only competes against the GPTQ_RESERVE_SECONDS budget, which is
negligible (~4s currently). We can afford 30s of AR generation effectively
free.

## Predicted outcome

Per PR #756: −0.0003 to −0.001 BPB. Small but legal and never tested in
eval-only context.

## Code change required

Add AR generation routine that runs before GPTQ collection:

```python
def collect_ar_self_gen_calib(model, vocab_size, n_batches, seq_len, device, temp=1.0):
    """Generate calibration data by autoregressive sampling from BOS token."""
    # 1. Start with BOS or random token
    # 2. Greedy/temperature sampling for n_batches × seq_len tokens
    # 3. Return as tensor for GPTQ Hessian collection
```

Wire into existing GPTQ calibration path with new env var:
- `GPTQ_CALIB_SOURCE` (default "val", new option "ar")
- `GPTQ_AR_TEMP` (default 1.0)

## Arms

| Arm | calib source | temp | batches | Notes |
|---|---|---|---|---|
| 046F-ar-temp1 | ar | 1.0 | 16 | direct PR #756 replication |
| 046F-ar-temp08 | ar | 0.8 | 16 | smoother distribution |
| 046F-ar-temp1-128 | ar | 1.0 | 128 | combine with 046A win |
| 046F-mixed | ar+val 50/50 | 1.0 | 16 | hedge |

## Predicted outcome

- 046F-ar-temp1: ~1.0744 (−0.0003 from baseline)
- 046F-ar-temp08: similar
- 046F-mixed: ~1.0742 (−0.0005)
- 046F-ar-temp1-128: depends on 046A outcome

## Cost

~30 min code + ~$4 for 4 arms = ~$5-6 total.

**Defer until after 046A (calib batches) lands** — knowing the optimal batch
count tells us how many AR samples to generate.
