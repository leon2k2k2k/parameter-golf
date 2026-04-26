# Loop Layer Improvements

**Status:** candidate — three related ideas, screen together
**Expected Δ:** +0.001 to +0.005 bpb combined; each lever is independent and stackable
**Sources:**
- Xu & Sato, "On Expressive Power of Looped Transformers" (ICML 2025, arXiv:2410.01405)
- LoopFormer (arXiv:2602.11451)
- Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"
- "On the Residual Scaling of Looped Transformers" (OpenReview 2026, id: bj8l2FYjSd)
- NL ablation series: specs 041H, 041K, 041N, 041L

## Prior art scan — openai/parameter-golf PRs (checked 2026-04-26)

| Lever | PRs found | Has result? |
|---|---|---|
| A — iteration embeddings | #1552 (additive pass embeds), #1554 (iteration_embed param), #1640 (FiLM γ/β per step) | **None** — all pending compute |
| B — MLP-only loop | None found | N/A — novel |
| C — residual 1/L init | None explicit; #1779 uses frozen learned alpha/beta (different) | Partial |

**Key finding for Lever A:** Three independent teams (#1552, #1554, #1640) have proposed iteration-level conditioning on this exact competition stack. All are OPEN with no training result. We would be the first to produce a number. The #1640 variant uses FiLM (scale + shift per step) which is strictly more expressive than additive embeddings alone — worth implementing both as arms.

**Key finding for Lever B:** No PR in the entire repo attempts MLP-only loop passes. Confirmed novel.

**Key finding for Lever C:** Our current baseline (#1779) already has `frozen_recurrent_alpha` — learned per-layer blend scalars trained to convergence then frozen. This partially addresses residual scaling (the alpha gate effectively learns a per-layer scale). Pure 1/L init on `attn_scale`/`mlp_scale` is still different and untested, but the marginal gain may be smaller than if we had no alpha at all. Lower priority vs A and B.

## Overview

Three improvements to the looped layers motivated by the 2025 recurrent-transformer literature. All three target the same architectural region (layers 4–5, Loop45) and are independent — any subset can be combined.

| Lever | Code change | Param cost | Risk | Expected Δ |
|---|---|---|---|---|
| A. Iteration embeddings | New `nn.Parameter` (zero-init) | ~1536 params | Low | +0.001 to +0.003 |
| B. MLP-only loop | Skip attn on passes 2+ | 0 | Medium | +0.001 to +0.004 |
| C. Residual 1/L init | Init-only, no new params | 0 | Very low | +0.001 to +0.002 |

---

## Lever A — Iteration Embeddings

### Idea

Our Loop45 baseline runs layers 4–5 identically on every pass. The looped block cannot distinguish pass 0 from pass 2. ICML 2025 proves this is a **fundamental approximation limitation**: naive weight-tied loops are representationally weaker than equivalently-sized non-looped networks, and timestep encoding closes the gap.

Fix: inject a small learned embedding per loop pass, so the block knows which iteration it is on.

```
Pass 0:  x = x + loop_embed[0];  run layers 4-5
Pass 1:  x = x + loop_embed[1];  run layers 4-5
Pass 2:  x = x + loop_embed[2];  run layers 4-5
```

`loop_embed` is `nn.Parameter(zeros(num_loops+1, model_dim))` — zero-init means step 0 is byte-identical to baseline. The block can specialize per pass: pass 0 does coarse routing, pass 1 does first refinement, pass 2 finalizes. LoopFormer and the Recurrent Transformer both independently validated per-step conditioning for weight-tied models.

**Parameter cost:** `(NL+1) × model_dim` ≈ 3 × 512 = 1536 params. Negligible.

### Code sketch

```python
# GPT.__init__
if loop_iter_embeds:
    self.loop_embeds = nn.Parameter(torch.zeros(num_loops + 1, model_dim))

# GPT.forward — inside loop schedule
for pass_idx, layer_idx in enumerate(loop_pass_schedule):
    if loop_iter_embeds and layer_idx == loop_start:
        x = x + self.loop_embeds[pass_idx]
    x = self.blocks[layer_idx](x)
```

Env var: `LOOP_ITER_EMBEDS=1` (default 0 → disabled).

### Risks
- Embeds may stay near zero if gradients are weak — log `loop_embeds.norm()` to verify they train.
- Zero-init means no effect on step 0, so there's no instability risk at launch.
- Unclear if any frontier PRs (#1797, #1756) already do this — check before spec freeze.

---

## Lever B — MLP-Only Loop

### Idea

Every loop pass currently runs the full block: attention then MLP. Hypothesis: **attention stabilizes on pass 0** (routing is established), while **MLP continues to refine** on passes 1+. Re-running attention on passes 2+ may be redundant re-routing to the same positions, costing FLOPs without adding information.

```
Pass 0:  attn(x) + MLP(x)   ← full block: establish routing + retrieve
Pass 1+: MLP(x) only        ← keep refining without re-routing
```

This is **unpublished** — every looped transformer paper (Universal Transformer, Looped Transformer, Recurrent Transformer, Hyperloop) loops the full block. MLP-only repeat passes are a genuinely novel ablation.

**Throughput payoff:** attention is O(seq²) + KV projections. Skipping it on passes 2+ is cheap enough that NL=4 or 5 MLP-only passes could cost less than NL=2 full passes. The NL=4 full-pass failure (spec 041N, −0.00374 regression) was throughput-collapse + over-attending pathology. MLP-only extra passes would avoid both.

**Theory:** Geva et al. 2021 show FFN layers act as key-value memories where each pass retrieves a different weighted mixture of facts. Multiple MLP applications on the same residual compose non-trivially — not the same as one larger MLP.

### Code sketch

```python
# Block.forward gains a loop_pass argument
def forward(self, x, loop_pass=0):
    if loop_pass == 0 or not mlp_only_loop:
        x = x + self.attn_scale * self.attn(self.norm1(x))
    x = x + self.mlp_scale * self.mlp(self.norm2(x))
    return x

# GPT.forward
for pass_idx, layer_idx in enumerate(loop_pass_schedule):
    pass_within_window = pass_idx // window_size
    x = self.blocks[layer_idx](x, loop_pass=pass_within_window)
```

Env var: `MLP_ONLY_LOOP=1` (default 0 → full block every pass).

Combines cleanly with Lever A — iteration embeddings still inject context on MLP-only passes.

### Risks
- Attention on passes 2+ may be doing real work: if MLP output substantially shifts the key/query space, re-attention carries real information. This is the main unknown.
- Gradient signal to attn weights comes only from pass 0. Attn weights may underfit in the looped layers — mitigated by the non-looped layers (0–3, 6–10) which still provide full gradient signal.
- `RECUR_ALPHA_ENABLED` interaction: if ever used, need to ensure alpha gate is compatible with MLP-only passes.

---

## Lever C — Residual 1/L Init

### Idea

When a block is repeated L times (weight-tied), each pass adds a full-scale residual. The 2026 OpenReview paper shows weight-tied looped blocks need **1/L residual scaling** for stable training — the standard 1.0 init overcooks the residuals by a factor of L.

Our baseline: `attn_scale = mlp_scale = nn.Parameter(ones(dim))` → initialized to 1.0 for all layers including the looped ones. With NL=2 (3 total passes through layers 4–5), layer 4 residuals accumulate 3× before the optimizer can compensate.

Fix: initialize `attn_scale` and `mlp_scale` for the looped layers to `1/L = 1/3` at NL=2. The parameters remain learned — this is purely an init change that gives the optimizer a better starting point.

```python
if layer_idx in loop_layer_indices:
    with torch.no_grad():
        block.attn_scale.fill_(1.0 / num_loop_passes)  # 1/3 at NL=2
        block.mlp_scale.fill_(1.0 / num_loop_passes)
```

Env var: `LOOP_SCALE_INIT=recip` (default: 1.0 as before).

This is the lowest-risk lever: no new params, no architecture change, just an init value. If it doesn't help, it costs nothing and leaves no debt.

### Risks
- At our budget (~5000 steps), the optimizer may correct the 1.0 init within 100–200 steps anyway, making this a no-op.
- Must be NL-dependent: init changes when NL changes (NL=3 → 1/4, NL=2 → 1/3). Tie to `NUM_LOOPS` env var.
- Apply only to looped layers (4–5). Non-looped layers keep 1.0 init.

---

## Combined screen plan

These three levers touch different parts of the loop forward pass and do not conflict. Recommended screen: one pod, multiple arms.

| Arm | Config | Purpose |
|---|---|---|
| Baseline | canonical 039b config | step-matched reference |
| A | `LOOP_ITER_EMBEDS=1` | iteration embeddings alone |
| B | `MLP_ONLY_LOOP=1` | MLP-only passes alone |
| C | `LOOP_SCALE_INIT=recip` | residual 1/L init alone |
| A+C | `LOOP_ITER_EMBEDS=1` + `LOOP_SCALE_INIT=recip` | natural stack (both low-risk) |
| B+C | `MLP_ONLY_LOOP=1` + `LOOP_SCALE_INIT=recip` | MLP-only with stable init |

Hardware: 2×H100, 4h 20min budget per arm (same as 039b baseline screen format).
Kill: ≥ 1.0670. Accept: ≤ 1.0641. Baseline: 039b at 1.06514.

If budget is tight, run A and B first (biggest expected signal), add C as a free rider on whichever wins.

Lever B (MLP-only loop) requires a code change → `exp/<slug>` branch + mini rung.
Levers A and C could technically be hyperparam-only if the embed is tiny enough, but the new `nn.Parameter` in A makes it a code change too. Single branch covers A + B + C.

## Stacking with other queued specs

- **044A/044B** (QK gain softening): fully orthogonal — different layers, different mechanism.
- **043A** (early loop activation): orthogonal — changes *when* looping starts, not *how* the loop runs. Can combine.
- **NL=3 (041K sweet spot)**: levers A and C stack cleanly onto NL=3. Lever B with NL=3 → even more MLP-only passes affordable.

## Open questions for spec freeze

1. Do any frontier PRs (#1797, #1756) already implement iteration embeddings? If yes, Lever A is less novel but still worth verifying on our stack.
2. Should the MLP-only loop skip attention on *both* layers 4 and 5 in passes 2+, or only on one of them?
3. Does `RECUR_ALPHA_ENABLED` need any adaptation for MLP-only passes?
