# Loop Iteration Embeddings

**Status:** candidate
**Expected Δ:** +0.001 to +0.003 bpb
**Source:** Xu & Sato, "On Expressive Power of Looped Transformers" (ICML 2025, arXiv:2410.01405); LoopFormer (arXiv:2602.11451)

## Core idea

Our Loop45 baseline repeats layers 4–5 identically on every pass — the looped block has no way to know whether it is on pass 1 or pass 2 or pass 3. ICML 2025 proves this is a fundamental approximation limitation: naive weight-tied loops cannot represent certain functions that a single deep network can. The fix is **timestep/iteration encoding**: before each loop pass, add a small learned embedding that signals the pass index to the block.

```
Pass 0:  x = x + loop_embed[0];  run layers 4-5
Pass 1:  x = x + loop_embed[1];  run layers 4-5
Pass 2:  x = x + loop_embed[2];  run layers 4-5
```

`loop_embed` is a learned matrix of shape `[num_loops+1, model_dim]` — initialized to zero so the first pass is byte-identical to baseline. This lets the looped block specialize per iteration: e.g., pass 0 is "coarse routing," pass 1 is "first-order refinement," pass 2 is "finalization."

## Why this is likely to help

1. **Theoretical guarantee**: Xu & Sato show timestep encoding closes the approximation gap. Without it, any looped transformer is representationally weaker than an equivalently-sized non-looped one. With it, the gap closes.
2. **Parameter cost is negligible**: `(num_loops+1) × model_dim` ≈ 3 × 512 = 1536 params on our stack — well under 0.001% of total budget.
3. **Zero-init is safe**: initializing `loop_embed` to zero means step 0 of training is byte-identical to baseline, so there is no risk of initialization instability.
4. **Multiple independent groups converged here**: LoopFormer, Recurrent Transformer, and ALBERT-with-layer-embeds all independently validated that per-step conditioning improves weight-tied models.

## Interaction with current baseline

We have `attn_scale` and `mlp_scale` (learned per-dim gated residuals) in the looped layers. These partially compensate, but they apply *identically* on every pass — they do not track which pass is running. Iteration embeddings add a complementary per-pass signal that `attn_scale`/`mlp_scale` cannot provide.

`RECUR_ALPHA_ENABLED` is a separate gating mechanism for the recurrence output blend; iteration embeddings do not conflict with it.

## Code-change sketch

In the `Block` module for loop layers 4–5 (or wherever `loop_start`/`loop_end` are active), add:

```python
# In GPT.__init__
if loop_iteration_embeds:
    self.loop_embeds = nn.Parameter(
        torch.zeros(num_loops + 1, model_dim)  # zero-init → safe
    )

# In GPT.forward, inside the loop:
for pass_idx, layer_idx in enumerate(loop_pass_schedule):
    if loop_iteration_embeds and layer_idx == loop_start:
        x = x + self.loop_embeds[pass_idx]
    x = self.blocks[layer_idx](x)
```

Env var: `LOOP_ITER_EMBEDS=1` (default 0 → disabled, byte-identical to baseline).

The embed is injected once per pass, at the entry to the loop window (layer 4). No change to the non-looped layers.

## Screen plan

Single-arm screen alongside any other loop-variant spec. This change is:
- Hyperparam only if we add a flag and the embed is small enough not to shift param count meaningfully
- Technically a code change (new parameter), so needs an `exp/<slug>` branch + mini rung

Screen on 2×H100 (4h 20min budget), compare vs canonical baseline (039b, pre-quant EMA 1.06514). Kill threshold: ≥ 1.0670. Accept: ≤ 1.0641.

## Risks / open questions

- **Does zero-init actually degenerate?** The embeds could stay near zero throughout training if the gradient through them is weak. Monitor `loop_embeds.norm()` in logs.
- **Injection point matters**: inject at entry to the loop window (before layer 4) vs. inside each block (before attn). Entry is simpler and avoids touching `Block` internals.
- **Interaction with NL=3**: spec 041K showed NL=3 is the sweet spot. Combine or screen separately? Recommend: screen NL=2 + embeds first, then NL=3 + embeds if positive.
- **Frontier PRs**: unclear if any top-10 PRs (#1797, #1756) implement something like this — worth checking before spec freeze.

## If this works

Stacks cleanly with early-loop activation (043A) and QK gain softening (044A/B) — none of those touch the per-pass conditioning. A winning result can be incorporated as a training-code change on `research` and used as baseline for subsequent specs.
