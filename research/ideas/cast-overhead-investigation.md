# Cast / dtype-conversion overhead — investigation thread

**Date opened:** 2026-04-29
**Status:** OPEN — needs more testing before any spec is justified.
**Trigger:** spec 064b chrome traces showed ~17% of rank-0 GPU kernel
time in kernels matching `cast|to_copy|convert`. Initial reading was
"wasted dtype overhead — easy throughput recovery." On closer
inspection that number is inflated by misclassification.

## What we actually saw

From `runs/064b-torch-profile/seed_42/chrome_trace_step_88.json`,
GPU kernel time breakdown (rank 0, 8-step active phase):

| Group | % of GPU time | Note |
|---|---|---|
| GEMM / matmul (cutlass + nvjet) | 24.5% | healthy |
| `linear_leaky_relu_square_kernel` | 12.8% | single activation kernel |
| **"cast/dtype" (regex group)** | **17.0%** | **misleading, see below** |
| RMSNorm / LayerNorm | 9.2% | |
| Elementwise | 4.9% | |
| NCCL | 0.5% | |
| Optimizer | 0.05% | |
| Misc / triton | ~31% | |

## Why the 17% is misleading

The "cast/dtype" group used regex `cast|to_copy|convert` (case-
insensitive). Looking at the actual kernels grouped under this label,
the dominant contributors are *fused Triton kernels* like:

- `triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_s`
- `triton_per_fused__to_copy__unsafe_view_add_empty_like_expand`
- `triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_emp`

`__to_copy__` shows up because torch.compile *fused* a dtype
conversion alongside many other ops. These kernels are doing real
composite work (RMSNorm backward, residual add, view manipulation),
not pure dtype conversion.

The actual "pure-cast" cost is probably 3-6%, not 17%. Need finer
classification to confirm.

## Sources of casts in the codebase (from reading train_gpt.py)

Explicit per-forward-pass casts on small fp32 parameters:

```python
# Block.forward (~line 1126)
mix = self.resid_mix.to(dtype=x.dtype)        # fp32 → bf16
x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(...)

# GPT._forward_hidden (~lines 1369-1389)
w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
```

Plus the parameter banks (`qo_bank`, `kv_bank`, `mlp_up_bank`,
`mlp_down_bank`) read in fp32 and used in bf16 matmuls.

## Three paths to consider, by risk

**Path 1 — refined classification (free, ~5 min).**
Re-analyze the existing chrome trace with a stricter regex that
separates pure casts from fused-with-cast. Gives us the actual
pure-cast number. Decision tree:
- pure-cast < 3% → close this thread, pivot
- pure-cast 5-10% → path 2 justified
- pure-cast > 10% → high priority

**Path 2 — store scale parameters in bf16 (medium risk).**
`attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`, `skip_gates`
are small tensors stored fp32 and cast to bf16 every forward. If
stored in bf16 directly, the cast disappears.

Risk: these are scale parameters; fp32 precision matters for
optimizer stats (Muon/Adam track them). Could destabilize training.
Bf16 has only ~7 bits of mantissa; fine-grained scale tuning may
exceed that resolution. Need a smoke run.

**Path 3 — tune torch.compile autocast (high risk, high effort).**
Set explicit dtype hints, audit autocast regions. Hard to predict
effect; touches compile config. Probably too risky for the homestretch
window; flag for a future session.

## What's known vs. unknown

| Known | Unknown |
|---|---|
| Some casts are happening every forward pass | What fraction of GPU time is pure cast |
| Top fused kernels include `__to_copy__` in name | Whether removing those casts would actually speed things up (torch.compile may already fuse them away) |
| Scale parameters are stored fp32 | Whether storing them bf16 destabilizes training |
| GEMMs are healthy at 24.5% | Whether the homestretch budget can absorb a smoke + retry cycle on this |

## Why this needs more testing before acting

1. **The "17% in casts" claim was wrong.** Acting on a misread number
   is bad protocol. Resolve it before speccing.
2. **The fix has nontrivial training-stability risk.** Storing scale
   params in bf16 could degrade the model in ways that don't show up
   until late in a 600s training run. We can't afford to discover
   that during the official rung.
3. **It's not even clear we have the budget for the diagnostic.**
   We've spent ~$11 on the 064 thread already (3× the spec budget).
   Spending more diagnostic budget on cast investigation may not
   pay off.

## Recommended next test (if we decide to chase this)

**Refined-grouping pass on the existing trace** — zero pod cost, runs
locally. Updated `analyze_chrome_trace.py` with three groups:

- `pure_cast`: kernel name matches `^.*to_copy$` or `^.*cast.*$` with
  no other op markers (e.g., not `_add_`, not `_mul_`, not `_view_`).
- `fused_with_cast`: kernel name includes `to_copy` AND other op
  markers. These do real work; not "wasted."
- `other`: everything else.

If pure_cast is <3%, this thread closes. If >5%, write a spec to test
storing one specific scale parameter (e.g., `resid_mix` — the smallest
and least optimizer-critical) in bf16 as a smoke test, then expand
based on result.

## Sibling investigation worth bundling

The squared-ReLU activation `linear_leaky_relu_square_kernel` is 13%
of GPU time as a single kernel. If torch.compile isn't fusing it into
the surrounding MLP matmul, we may be writing/reading the activation
output to/from HBM unnecessarily. Worth checking in the same trace
analysis pass — could be a bigger win than the cast story.

## Decision pending

Before any spec is frozen on this:
1. Run the refined-grouping analysis (free).
2. If signal exists, scope a minimal-risk smoke spec.
3. Compare expected throughput recovery (likely 2-5%, not 17%) vs
   homestretch opportunity cost (other untested ideas: EMA decay
   sweep, baseline #1851 port, multi-seed 060A).
