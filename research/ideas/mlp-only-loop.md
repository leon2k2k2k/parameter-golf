# MLP-Only Loop

**Status:** candidate — novel (no prior art found in literature)
**Expected Δ:** +0.001 to +0.004 bpb if MLP is the bottleneck; 0 or negative if attention re-routing matters on repeat passes
**Source:** internal reasoning from Geva et al. 2021 ("FFNs are key-value memories") + NL series ablations (041H, 041K, 041N)

## Core idea

Our Loop45 baseline repeats the full block (attention + MLP) on every loop pass. The hypothesis here is that **attention is primarily a routing/position mechanism** that stabilizes on the first pass, while **MLP is an associative memory retrieval** that benefits from repeated application. On passes 2+, re-running attention may be redundant (re-routing to the same positions) while re-running MLP continues to refine.

Proposed variant:

```
Pass 0:  full block  →  attn(x) + MLP(x)   (routing + retrieval)
Pass 1+: MLP only    →  MLP(x)              (further retrieval only)
```

This is **unexplored in published literature** — all looped transformer papers (Universal Transformer, Looped Transformer, Recurrent Transformer, Hyperloop) repeat the full block. MLP-only repeat passes would be a novel ablation.

## Why this might help

1. **Throughput gain**: attention is expensive — O(seq²) plus KV projections. Skipping attn on passes 2+ reduces the per-loop-pass cost. With NL=2 baseline (3 total passes), we could go to NL=4 or 5 (5–6 passes, but only one attention run) for equivalent or lower FLOP cost. More effective refinement steps for free.

2. **NL=4 failure is consistent with this**: spec 041N showed NL=4 (4 full attention passes) regresses badly — both throughput collapse (~20%) and quality degradation. The throughput collapse is cache-eviction-driven; the quality collapse may be "over-attending" pathology. MLP-only extra passes would avoid both failure modes.

3. **Geva et al. 2021**: FFN layers act as key-value memories where each application retrieves a different weighted mixture of stored facts. Multiple MLP applications on the same residual state are non-trivially composable — not the same as one larger MLP. This is the primary theoretical motivation.

4. **Attention convergence**: mechanistic interpretability work shows attention patterns often converge after 1–2 forward passes when inputs change slowly. In a looped architecture where the residual is incrementally refined (not radically different), pass-2 attention patterns likely mirror pass-1 closely.

## Interaction with current baseline

The looped layers are 4–5 (loop_start=3, loop_end=5 in 0-indexed config). Each block has:
- `self.attn` (multi-head attention with RoPE)
- `self.mlp` (gated MLP with leaky_relu_square)
- `self.attn_scale`, `self.mlp_scale` (learned gated residuals)
- `RECUR_ALPHA_ENABLED` (optional residual blend gate — currently off by default)

MLP-only passes would skip `self.attn` and its gated residual, only applying `self.mlp`. The `attn_scale` and `mlp_scale` learned parameters still apply on the MLP pass (so the gated residual is preserved, just attn output is zero).

## Code-change sketch

```python
# In Block.forward, add `loop_pass` argument:
def forward(self, x, loop_pass=0, mlp_only_from=1):
    if loop_pass < mlp_only_from:
        # full block
        x = x + self.attn_scale * self.attn(self.norm1(x))
    x = x + self.mlp_scale * self.mlp(self.norm2(x))
    return x

# In GPT.forward loop:
for pass_idx, layer_idx in enumerate(loop_pass_schedule):
    pass_within_window = pass_idx // window_size  # which loop pass
    x = self.blocks[layer_idx](x, loop_pass=pass_within_window,
                                mlp_only_from=mlp_only_from)
```

Env var: `MLP_ONLY_FROM=1` (default 0 → never, full block every pass; 1 → MLP-only from pass 2 onward).

Can be combined with `LOOP_ITER_EMBEDS` — iteration embeddings still inject context even on MLP-only passes.

## Screen plan

2×H100, 4h 20min budget. Test `MLP_ONLY_FROM=1` (pass 2+ are MLP-only) with NL=2 (baseline loop count) and also with NL=3 or NL=4 (since attn is cheaper now, more passes are affordable).

Suggested arms:
- A: `MLP_ONLY_FROM=1`, `NUM_LOOPS=2` (same pass count, skip attn on pass 2)
- B: `MLP_ONLY_FROM=1`, `NUM_LOOPS=3` (add one extra MLP-only pass at low cost)

Compare vs canonical baseline (039b, 1.06514). Kill ≥ 1.0670. Accept ≤ 1.0641.

## Risks / open questions

- **Attention in pass 2+ might be doing real work**: if the residual changes enough between passes (e.g., MLP output substantially shifts the key/query space), re-attention is meaningful. This is the main risk — unknown without experiment.
- **`recur_alpha` interaction**: if `RECUR_ALPHA_ENABLED=1` is ever used, the alpha gate blends between looped and skip-connection. MLP-only passes need a compatible alpha gate (or skip alpha for MLP-only passes).
- **Norm interaction**: the MLP reads `self.norm2(x)`. On MLP-only passes, `norm2` still normalizes the residual — this is fine and unchanged from baseline behavior.
- **Gradient flow**: skipping attn on passes 2+ means no gradient flows back through attn weights from those passes. Effective gradient signal for attn weights comes only from pass 1. This might underfit attention in the long run, or it might be fine since attention is well-trained from non-looped layers too.

## If this works

High novelty — no paper has published this ablation. A positive result would be a genuine contribution beyond the competition, and would justify further research into hybrid per-pass scheduling (e.g., attn on passes 1 and 3 only, MLP on all passes). Could combine with iteration embeddings (which would give MLP-only passes their own step-conditioning).
