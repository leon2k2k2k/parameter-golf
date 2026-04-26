# Loop Recurrence Normalization

**Status:** candidate — three ideas, motivated by Lever C's success in spec 045
**Expected Δ:** −0.001 to −0.004 per lever
**Born from:** spec 045 eval (2026-04-26). Lever C (1/L init) showed consistent −0.003 to −0.010 train loss advantage from step 400 through end of run, with largest gain at loop activation (−0.015 at step 2300). This confirms residual magnitude across loop passes is load-bearing. These ideas deepen that attack.

## Background

In Loop45, layers 4–5 are weight-tied and run 3 total passes. Each pass adds:
```
x = x + attn_scale * attn_output
x = x + mlp_scale * mlp_output
```

`attn_scale` and `mlp_scale` are shape `[model_dim]` parameters, **shared across all passes**
(weight-tied). Lever C's insight: initializing them to 1/3 instead of 1.0 prevents the 3×
residual accumulation imbalance at training start. The optimizer still adjusts them, but from
a calibrated starting point.

Three logical extensions:

---

## Idea 1 — Per-Pass Residual Scales

### Motivation
Lever C is a static init fix. The real question is: should pass 0, pass 1, and pass 2 have
the *same* attn/mlp scale at convergence? Almost certainly not — pass 0 does coarse routing,
pass 1 refines, pass 2 finalizes. Making the scales per-pass allows the model to discover
the right contribution magnitude at each depth level.

### Implementation
Replace the shared `[model_dim]` attn_scale/mlp_scale for looped layers with per-pass
`[num_passes, model_dim]` tensors:

```python
# GPT.__init__: for looped layers, create per-pass scale params
if h.loop_per_pass_scales and h.num_loops > 0:
    num_passes = h.num_loops + 1
    self.loop_attn_scales = nn.Parameter(
        torch.full((num_passes, h.model_dim), 1.0 / num_passes)
    )  # init to 1/L same as Lever C
    self.loop_mlp_scales = nn.Parameter(
        torch.full((num_passes, h.model_dim), 1.0 / num_passes)
    )

# _forward_hidden: inject per-pass scale in block call
pass_attn_scale = self.loop_attn_scales[pass_idx]  # [model_dim]
pass_mlp_scale  = self.loop_mlp_scales[pass_idx]
x_new = self.blocks[i](x, x0, ..., loop_attn_scale=pass_attn_scale, loop_mlp_scale=pass_mlp_scale)

# Block.forward: use injected scale if provided, else self.attn_scale
```

**Param cost:** 2 × 3 × 512 = 3072 params. Negligible.
**Env var:** `LOOP_PER_PASS_SCALES=1`
**Risk:** Medium. Requires overriding attn_scale/mlp_scale inside Block.forward for looped layers only. Need to ensure non-looped layers still use their own per-layer scales.
**Novel:** No prior art in ~300 PR scan.

### Relationship to Lever A / Lever C
- Lever A: additive per-pass *bias* to x at loop entry (shifts hidden state)
- Lever C: per-layer scale init (static, shared across passes)
- Idea 1: per-pass *multiplicative* scale of residual (learned, dynamic)
These are orthogonal and stackable.

---

## Idea 2 — Inter-Pass RMSNorm

### Motivation
After each complete pass through layers 4–5, the hidden state `x` has accumulated
attn + MLP residuals. Before the next pass, normalizing `x` back to unit norm prevents
the residual from compounding in unpredictable ways. Standard pre-norm inside the block
already normalizes `x_in = mix * x + (1-mix) * x0` before attention and MLP, but this is
norm-then-project, not norm-of-the-full-residual-stream.

### Implementation
Insert a lightweight RMSNorm (no learnable params) between each loop pass:

```python
# _forward_hidden: after completing a full window pass (both layers 4 and 5),
# before starting the next pass
if layer_idx == loop_end and pass_idx < num_passes - 1 and h.inter_pass_norm:
    x = F.rms_norm(x, (x.size(-1),))
```

Optionally with a learned per-pass scale (shape `[num_passes-1, model_dim]`):
```python
    x = F.rms_norm(x, (x.size(-1),)) * self.inter_pass_norm_scale[pass_idx]
```

**Param cost:** 0 (unlearned) or 2 × 512 = 1024 (with per-pass scale). 
**Env var:** `INTER_PASS_NORM=1`
**Risk:** Low-medium. Adds one norm op per inter-pass boundary (2 norms for NL=2). May interfere with residual learning — the optimizer expects to be able to control residual magnitude through attn_scale/mlp_scale, and an explicit renorm overrides this. Test unlearned version first.

### Note on pre-norm interaction
The blocks already pre-norm inside (`attn_norm(x_in)`, `mlp_norm(x_out)`). So the
inter-pass norm operates on the RESIDUAL STREAM before it's mixed with x0 for the next pass.
Not redundant — pre-norm normalizes the *input to attention/MLP*, inter-pass norm
normalizes the *hidden state across passes*.

---

## Idea 3 — FiLM Per-Pass Conditioning

### Motivation
PR #1640 (no results yet) proposed learned scale+shift per loop step (FiLM). This is
strictly more expressive than Lever A's additive embeds (Lever A is FiLM with γ=1):
```
x = γ[pass] * x + β[pass]
```

FiLM lets the model rescale the hidden state magnitude per pass, not just shift it.
This combines the intent of Idea 1 (per-pass scale) with Lever A (per-pass shift)
in a simpler architecture: one transform to the whole residual stream, not separate
attn/mlp scale overrides.

### Implementation
```python
# GPT.__init__
if h.loop_film_cond and h.num_loops > 0:
    num_passes = h.num_loops + 1
    # gamma: init to ones (identity scale), beta: init to zeros (no shift)
    self.loop_film_gamma = nn.Parameter(torch.ones(num_passes, h.model_dim))
    self.loop_film_beta  = nn.Parameter(torch.zeros(num_passes, h.model_dim))

# _forward_hidden: at loop_start entry per pass
if h.loop_film_cond and self.looping_active and layer_idx == loop_start:
    g = self.loop_film_gamma[pass_idx].to(dtype=x.dtype)
    b = self.loop_film_beta[pass_idx].to(dtype=x.dtype)
    x = g * x + b
```

**Param cost:** 2 × 3 × 512 = 3072 params.
**Env var:** `LOOP_FILM_COND=1`
**Risk:** Medium. The gamma=ones init means no effect at step 0 (safe). But multiplicative
conditioning on the full hidden state could cause instability if gamma drifts large.
Gradient monitoring needed (similar to Lever A's loop_embeds.norm() logging).

### Relationship to Lever A
FiLM strictly generalizes Lever A. If we spec this, Lever A becomes the "beta-only ablation"
and FiLM becomes the full version. Prior art: PR #1640 (pending), LoopFormer paper.

---

## Screen plan

If Arm D (C alone) wins:
1. Run **Idea 1** (per-pass scales) stacked on D — tests whether dynamic per-pass
   calibration beats static 1/L init
2. Run **Idea 3** (FiLM) — tests whether combined scale+shift beats additive-only Lever A

If Arm D is in noise zone:
1. Run **Idea 2** (inter-pass RMSNorm) — very cheap, independent idea
2. Consider combining C + Idea 1 + Idea 2 as a stack

Inter-pass RMSNorm (Idea 2) is compatible with all other levers and can be added as an
extra arm to any future screen.

## Stacking with queued specs

- **Arm B (MLP-only loop)**: Idea 1 and Idea 2 are orthogonal to whether attention is
  skipped on later passes. All three can stack.
- **044A/044B (QK gain softening)**: fully orthogonal — different layers and mechanism.
- **043A (earlier loop activation)**: orthogonal to normalization approach.
