# Spec 053 — Per-Pass FFN Output Scale

**Date:** 2026-04-28
**Branch:** `exp/052-perpass-ffn-scale` @ `cc64dd4`
**Note:** branch named 052 because spec 052 slot was taken (PPM-port-tuned); spec number is 053.

## Hypothesis

Adding a learned `[n_passes, n_loop_layers, model_dim]` scale vector on the MLP output of each loop layer — initialized to ones — lets each recurrence pass specialize its MLP contribution with near-zero parameter cost. Based on LoopFormer finding: pass-phase signals give +1.93pp zero-shot at near-zero cost. Transparent at init → can only help if gradient finds signal.

## Baseline

Spec 050A (`exp/050-ac-fix-on-1797`), pre-quant EMA bpb **1.06484**.

## Expected Δ

Small: −0.001 to −0.003. Low confidence (9 params, 4608 floats total).

## Accept criteria

Pre-quant EMA bpb ≤ **1.064** (Δ ≤ −0.00084). Kill if > 1.068.

## Config diff

No hyperparameter changes — pure architecture addition.

```
# New: per-pass FFN scale (ones-init)
LOOP_PASS_FFN_SCALE=1   # env var not used; controlled by code
# New params: 3 passes × 3 loop layers × 512 = 4,608
```

## Code changes

**Branch:** `exp/052-perpass-ffn-scale` @ `cc64dd4`

Key diff:
- `GPT.__init__`: `loop_pass_ffn_scale = nn.Parameter(torch.ones(n_passes, n_loop, model_dim))`; `_ffn_pass_scale_id` ones buffer for non-loop layers
- `_forward_hidden`: enc/dec loops enumerate steps, look up (pass, layer) info, pass `ffn_pass_scale` tensor (learned or identity buffer) to each block call
- `Block.forward` + `_parallel_block`: accept `ffn_pass_scale=None`; applies `mlp_out = scale[None,None,:] * mlp_out` when not None
- Always-tensor: post-loop graph always gets a tensor (never None) → single compiled variant per regime

## Hardware ladder

- **2×H100 mini (20-min screen):** DONE — see result below
- **8×H100 official:** pending decision based on screen result

## Result (screen run)

Pre-quant EMA bpb: **1.06440** vs baseline 1.06484 → **Δ = −0.00044**
Submission size: baseline + 1,169 bytes. ~825 KB headroom remaining.

Decision: within noise (~0.5× SOTA std). Marginal positive. Promote to official only if combined with 054.

## Seed plan

Single seed (42) for screen; 3 seeds for official if promoted.

## Inputs

- Data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...`
- Tokenizer: `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- Hotstart: none (train from scratch, same as 050A)

## Checkpoints to emit

- `final_model.pt` (EMA weights)
- `final_model.int6.ptz` (quantized submission)

## Stop-early criteria

NaN loss, val_bpb > 1.075, step time > 2× baseline.

## Cost estimate

~$3 per 20-min 4×H100 screen.
