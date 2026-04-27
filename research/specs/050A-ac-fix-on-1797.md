# Spec 050A — AC-fix levers on 1797 baseline

**Date:** 2026-04-27
**Branch:** `exp/050A-ac-fix-on-1797` @ `62a0c0e`
**Launch script:** `tmp_exec/launch_050A_ac_fix.sh`

## Hypothesis

Recip-init + iter-embeds + re-warmup ported to the cleaner 1797 base. Recip-init keeps residual magnitude stable across N loop passes; iter-embeds give each pass a distinct identity signal; re-warmup prevents mid-run compile hang at loop activation.

## Baseline

Spec 050 (`f5b8af8`) — pre-quant EMA val_bpb ~1.067.

## Expected Δ

−0.001 to −0.003 pre-quant. Low confidence.

## Accept criteria

- Pre-quant EMA val_bpb ≤ 1.067.
- `loop_rewarm:` line appears in train.log (re-warmup fired).

## Config diff (vs spec 050)

```
LOOP_SCALE_INIT=recip
LOOP_ITER_EMBEDS=1
```

## Code changes

Branch: `exp/050A-ac-fix-on-1797` @ `62a0c0e`
Train script: `records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py`

- `GPT.__init__`: recip-scale loop layer attn/mlp scales; `loop_iter_embeds` param `[passes, hidden]`
- `_forward_hidden` enc/dec loops: `x += loop_iter_embeds[pass_idx]` at loop entry
- Training loop: `_run_cu_bucket_warmup()` immediately after `looping_active = True`

## Hardware ladder

- **4×H100 mini** — straight to 20-min screen (no new compiled variants vs base loop).
- **8×H100 official** — if mini shows clear improvement.

## Seed plan

Single seed (42).

## Inputs

`/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...` No hotstart.

## Checkpoints

`final_model.pt` from GPTQ.

## Stop-early criteria

NaN, train_loss > 7.0 after step 200, step time > 1.5× baseline.

## Cost estimate

~$2 (20-min screen, 4×H100).

## Open questions for interview

Verify `loop_rewarm:` appears in train.log after loop activation.
