# Spec 046F — Short-burst AR self-generated calibration (Tier 2)

**Slug:** `046F-ar-selfgen-calib`
**Created:** 2026-04-27
**Status:** READY (code landed in commit 381baf2)
**Branch:** `exp/046-quant-repair`
**Commit:** `381baf2`
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

## Code (landed in 381baf2)

`ARSelfGenCalibLoader` class in train_gpt.py (line 2608):
- Drop-in replacement for `ShuffledSequenceLoader` at the GPTQ calib site
- `next_batch(global_tokens, grad_accum_steps)` returns (x, y) like upstream
- Pre-generates `n_batches` of AR samples at construction; serves from cache
  to avoid generation overhead per `next_batch` call
- AR generation: starts each sequence with BOS token, samples one token at
  a time using `base_model.forward_logits(x[:, :t])` + temperature softmax + multinomial
- Generation uses `torch.no_grad()` + bf16 autocast for speed

Wired into the calib_loader instantiation in `serialize()` at line 3375:
```python
if h.gptq_calib_source == "ar":
    calib_loader = ARSelfGenCalibLoader(...)
else:
    calib_loader = ShuffledSequenceLoader(h, device)
```

Env vars (default OFF):
- `GPTQ_CALIB_SOURCE="ar"` to activate (default "train" preserves baseline)
- `GPTQ_AR_TEMP` sampling temperature (default 1.0, neutral)
- `GPTQ_AR_SEQ_LEN` per-sample length (default 512 vs train_seq_len 2048;
  shorter to bound generation time — 16 batches × 512 ≈ 8K AR forward steps)

**Performance note**: AR generation is sequential (one token at a time),
significantly slower than disk reads. At seq_len=512 × n_batches=16 with
device_batch_size~16, generation takes ~1-3 min before GPTQ even starts.
Acceptable in eval-only mode but unaffordable during training (which is
why PR #1234 failed).

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

## Launch form (per arm)

Same as 046 verification spec, plus:

```bash
# 046F-ar-temp1 (PR #756 replication)
export GPTQ_CALIB_SOURCE=ar
export GPTQ_AR_TEMP=1.0
export GPTQ_AR_SEQ_LEN=512
export RUN_ID="046F-ar-temp1"

# 046F-ar-temp08 (smoother distribution)
export GPTQ_AR_TEMP=0.8
export RUN_ID="046F-ar-temp08"

# 046F-ar-temp1-len2048 (full seq_len — slow, last)
export GPTQ_AR_TEMP=1.0
export GPTQ_AR_SEQ_LEN=2048
export RUN_ID="046F-ar-fullseq"
```

Plus standard armD env vars and RESUME_FROM_CKPT.

## Acceptance

Reference = 046 verification quantized (1.07467).

- **Win**: < 1.0735
- **Noise**: 1.0735–1.0760
- **Kill**: > 1.0760

Per PR #756 the expected gain is small (-0.0003 to -0.001). Most likely outcome:
all arms in low-noise zone with possibly a +small or -small drift. Worth running
for the data point + to confirm AR self-gen works on this stack with current
LQER + CaseOps.

## Cost

~$1-2 per arm × 3 arms = ~$3-6 total.

**Defer until after 046A (calib batches) lands** — knowing the optimal batch
count tells us how many AR samples to generate. If 046A shows 32 vs 128 doesn't
matter, AR seq_len/batch tuning probably doesn't matter much either.
