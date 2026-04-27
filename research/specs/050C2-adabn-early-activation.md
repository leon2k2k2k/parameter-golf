# Spec 050C2 — Per-pass AdaLN with earlier loop activation (0.25)

**Date:** 2026-04-28
**Branch:** `exp/050C-adabn-on-1797` @ `cd74fcc`
**Launch script:** `tmp_exec/launch_050C2_adabn_early.sh`

## Hypothesis

Same AdaLN conditioning as 050C but loop activates at 25% of wallclock instead of 35%. Gives AdaLN γ/β params ~50% more post-loop training steps (~3700 vs ~2800 in a 20-min run). AdaLN starts near identity (γ=1, β=0) and needs gradient steps to diverge — earlier activation gives it more time to learn.

## Baseline

Spec 050 (`f5b8af8`) — pre-quant EMA val_bpb ~1.06438.

## Expected Δ

Uncertain. If 050C shows signal, 050C2 should be equal or better. If AdaLN needs more steps to show effect, earlier activation could unlock it.

## Accept criteria

- Pre-quant EMA val_bpb ≤ 1.067.

## Config diff (vs spec 050C)

```
ENABLE_LOOPING_AT=0.25   # was 0.35
```

Everything else identical to 050C.

## Code changes

No new code. Same branch/commit as 050C: `exp/050C-adabn-on-1797` @ `cd74fcc`.

## Hardware ladder

- **4×H100 mini** — straight to 20-min screen. No smoke needed if 050C's inductor cache stash exists at `/workspace/.inductor_cache_cd74fcc_050C` (same code, same graph variants). Launch script restores from stash or re-runs smoke if stash missing.
- **8×H100 official** — if mini shows clear improvement.

## Seed plan

Single seed (42).

## Inputs

`/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...` No hotstart.

## Checkpoints

`final_model.pt` from GPTQ.

## Stop-early criteria

NaN, train_loss > 7.0 after step 200, step time > 1.5× baseline at loop activation.

## Cost estimate

~$2 (20-min screen, 4×H100). Cache restore from 050C smoke saves ~$1 vs fresh smoke.

## Open questions for interview

- Does loop activate cleanly at ~step 1250 (vs ~1700 for 050C)? Verify `layer_loop:enabled` appears earlier.
- Any throughput difference vs 050C at loop activation?
