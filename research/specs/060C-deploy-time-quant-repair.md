# Spec 060C — 046L deploy-time quant repair on 060A baseline

**Date:** 2026-04-29
**Branch:** `exp/060C-deploy-repair` (forked from research, code from `exp/046-quant-repair` @ `fcb816f`)
**Parent:** 060A checkpoint + 046L code (already exists, just needs cherry-pick into 060A's train_gpt.py).

## Hypothesis

046L's deploy-time quant repair runs a passthrough fp16 fit AT EVAL TIME using AR-self-generated calibration data. Costs ZERO bytes (uses spare 100-180s of eval budget), bypasses the 16MB cap entirely. On 045-armD it was specced but never fully measured; predicted gain ~−0.001 to −0.005 BPB if it acts even partially like TTT.

## Baseline

060A.

## Expected Δ

**−0.001 to −0.005 BPB**, low confidence (was never cleanly validated end-to-end, only specced).

## Accept criteria

- post-quant + post-TTT val_bpb ≤ (060A − 0.0005)
- eval_time still ≤ 600s (deploy-time repair runs in ~60s; should fit)
- no NaN, no instability

## Config diff vs 060A

```
DEPLOY_TIME_REPAIR_ENABLED=1
DEPLOY_TIME_REPAIR_BATCHES=8
DEPLOY_TIME_REPAIR_SEQ_LEN=512
DEPLOY_TIME_REPAIR_LR=1e-3
DEPLOY_TIME_REPAIR_ITERS=5
```

## Code changes

Cherry-pick three blocks from `exp/046-quant-repair` @ `fcb816f` into 060A's `train_gpt.py`:

1. `fit_passthrough_to_self_consistency()` — the passthrough param fit function (~50 lines).
2. `ARSelfGenCalibLoader` class — generates AR samples without val data leak (~30 lines).
3. Hook in `train_and_eval()` after `deserialize()` (~10 lines):
   ```python
   eval_model = deserialize(h, device)
   if h.num_loops > 0:
       eval_model.looping_active = True
   if h.deploy_time_repair_enabled:
       repair_calib = generate_ar_calib(eval_model, h, n_batches=h.deploy_time_repair_batches, seq_len=h.deploy_time_repair_seq_len)
       fit_passthrough_to_self_consistency(eval_model, repair_calib, h)
   ```

Plus 5 new env-var-driven Hyperparameters fields.

## Hardware ladder

- 4×H100, RESUME_FROM_CKPT mode (no re-train; load 060A's pt, repair, eval).
- ~10-15 min wall, ~$3 cost.

## Seed plan

1 seed: 42.

## Inputs

- Hotstart: `/workspace/runs/060A-1855-port/seed_42/final_model.pt`
- AR calib generated at eval time (no external data needed beyond what's in the model).

## Stop-early criteria

- Repair fit diverges (loss > initial × 2 after 3 iters) → skip repair, run vanilla eval
- Post-repair val_bpb > 1.075 → kill (repair broke the model)

## Cost estimate

~$3, ~15 min wall.

## Open questions

1. Cherry-pick onto 060A's #1855-derived train_gpt.py — verify the LR schedule + LoRA-A path doesn't conflict with the 046L hooks.
2. AR-calib generator may need adjustment for #1855's slightly different forward signature.
