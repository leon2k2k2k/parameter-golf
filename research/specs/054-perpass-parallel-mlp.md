# Spec 054 — Per-Pass Attention Temperature + Parallel Bottleneck MLP

**Date:** 2026-04-28
**Branch:** `exp/053-perpass-parallel-mlp` @ `92c8085`
**Note:** branch named 053 because numbering shift from PPM-port collision; spec number is 054.

## Hypothesis

Loop recurrence is dominated by tied weights. Two complementary levers (both transparent at init) should let each pass specialize without recompile risk:
1. **Per-pass attention temperature** (`loop_pass_attn_temp [3,3]`, ones-init): scale QK dot products per (pass, layer) → each pass can sharpen/diffuse attention independently.
2. **Parallel bottleneck MLP** (`loop_pass_par_up [3,3,32,512]` kaiming, `loop_pass_par_down [3,3,512,32]` zeros-init): dim→32→dim residual added to main MLP output — zeros-init par_down makes it transparent at init; bottleneck=32 keeps K≥16 for Triton.

ALBERT finding: FFN tying is the dominant quality cost in depth-recurrent transformers; attention tying is approximately free. This targets the FFN gap with a capacity-additive approach (not a replacement).

## Baseline

Spec 050A (`exp/050-ac-fix-on-1797`), pre-quant EMA bpb **1.06484**.

## Expected Δ

Moderate: −0.002 to −0.005. Medium confidence. The parallel bottleneck is ~295K learnable params with a clear gradient path; the attn_temp is a soft probe for pass-phase signals.

## Accept criteria

Pre-quant EMA bpb ≤ **1.064** (Δ ≤ −0.00084). Kill if > 1.068.

## Config diff

No hyperparameter changes — pure architecture addition.

```
# New params: 9 (attn_temp) + 147,456 (par_up) + 147,456 (par_down) = 294,921
# Estimated quantized size: ~216 KB additional; baseline headroom ~825 KB
```

## Code changes

**Branch:** `exp/053-perpass-parallel-mlp` @ `92c8085`

Key diff:
- `GPT.__init__`: `loop_pass_attn_temp [3,3]` ones; `loop_pass_par_up [3,3,32,512]` kaiming; `loop_pass_par_down [3,3,512,32]` zeros; `_attn_temp_id` ones(1) and `_par_up_id/par_down_id` zeros identity buffers
- `_forward_hidden`: enc/dec loops look up (pass, layer) info per step; loop layers get learned tensors, non-loop layers get identity buffers → always-tensor pattern, no None/Tensor mix within post-loop graph
- `CausalSelfAttention.forward`: `if attn_temp is not None: q = q * attn_temp.to(dtype=q.dtype)` (broadcasts scalar [1] over [B,T,H,D])
- `Block.forward` + `_parallel_block`: `if par_down is not None:` applies parallel bottleneck: `par_h = gelu(linear(x_normed, par_up)); mlp_out += linear(par_h, par_down)`

**Compile safety:**
- Pre-loop graph: all None → branches never taken → single variant ✓
- Post-loop graph: all Tensor → branches always taken → single variant ✓
- No mid-run recompile risk

## Hardware ladder

- **2×H100 mini (20-min screen):** launch script at `tmp_exec/launch_053_perpass_parallel_mlp.sh` (note: script named 053, run dir is `runs/054-perpass-parallel-mlp-screen` — update on execution)
- **8×H100 official:** pending screen result

## Seed plan

Single seed (42) for screen; 3 seeds for official if promoted.

## Inputs

- Data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...`
- Tokenizer: `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- Hotstart: none (train from scratch)

## Checkpoints to emit

- `final_model.pt` (EMA weights)
- `final_model.int6.ptz` (quantized submission)

## Stop-early criteria

NaN loss, val_bpb > 1.075, step time > 2× baseline (par bottleneck should add <1% overhead).

## Cost estimate

~$3 per 20-min 4×H100 screen + ~$6 prewarm (new compiled graph).

## Open questions for interview

1. Run dir should be `runs/054-perpass-parallel-mlp-screen` — update the launch script's RUNDIR before executing.
2. Confirm RUNDIR in launch script matches this spec number.
