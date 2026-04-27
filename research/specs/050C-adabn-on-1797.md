# Spec 050C — Per-pass AdaLN on 1797 baseline (isolated)

**Date:** 2026-04-27
**Branch:** `exp/050C-adabn-on-1797` @ `cd74fcc`
**Launch script:** `tmp_exec/launch_050C_adabn.sh`

## Hypothesis

Per-pass (γ, β) conditioning on attn and MLP residual branches, tested in isolation vs 050 base (no iter-embeds, no recip-init). Plain elementwise ops — natively traceable by dynamo, no `@allow_in_graph` needed. Init: γ=1, β=0 (transparent). Clean retry of 047D on a simpler baseline.

## Baseline

Spec 050 (`f5b8af8`) — pre-quant EMA val_bpb ~1.067.

## Expected Δ

−0.001 to −0.003 pre-quant. Uncertain — 047D was inconclusive due to compile issues; this is a clean test.

## Accept criteria

- Pre-quant EMA val_bpb ≤ 1.067.
- Smoke log shows `layer_loop:enabled` and `loop_rewarm` — no hang at loop activation.

## Config diff (vs spec 050)

```
LOOP_ADABN=1
# LOOP_ITER_EMBEDS and LOOP_SCALE_INIT left at defaults (0 / "ones") — isolate AdaLN signal
```

## Code changes

Branch: `exp/050C-adabn-on-1797` @ `cd74fcc`
Train script: `records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py`

- `_adabn_apply(x, γ, β)`: standalone `γ*x+β` function before Block class
- `Block.forward`: optional `pass_γ_attn`, `pass_β_attn`, `pass_γ_mlp`, `pass_β_mlp` kwargs; applied after attn/mlp scaling before residual add
- `GPT.__init__`: `loop_adabn_γ/β_attn/mlp` params `[passes, loop_layers, hidden]`; ones/zeros init
- `GPT.__init__`: registers `_adabn_γ_id`/`_adabn_β_id` identity buffers (ones/zeros, shape [hidden])
- `_forward_hidden` enc/dec loops: always passes explicit `pass_γ/β_attn/mlp` kwargs — learned tensors for loop-layer steps, identity buffers for non-loop steps; no `**dict` expansion; single `Block.forward` compiled graph variant throughout (no mid-run recompile at loop activation)

## Hardware ladder

- **4×H100 mini** — 10-min smoke then 20-min screen (smoke required: new None→tensor graph variant).
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

~$3 (10-min smoke + 20-min screen, 4×H100).

## Open questions for interview

Verify smoke log shows `layer_loop:enabled` and `loop_rewarm`. Watch step time at loop activation.
