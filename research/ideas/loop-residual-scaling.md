# Loop Residual Scaling (1/L init)

**Status:** candidate — low risk, stackable with anything
**Expected Δ:** +0.0005 to +0.002 bpb (stability improvement; not guaranteed to show at this scale)
**Source:** "On the Residual Scaling of Looped Transformers" (OpenReview 2026, forum id: bj8l2FYjSd)

## Core idea

When a block with residual connections is repeated L times (weight-tied), each pass adds a full-scale residual contribution. The total residual addition is O(L) times larger than in a non-looped block. Standard deep transformer initialization (from Vaswani et al.) targets a specific activation variance at each layer; looping breaks this — activations can grow as O(√L) or O(L) depending on the initialization.

The 2026 paper finds: **weight-tied looped blocks require 1/L residual scaling for stable training**, where L = total number of passes through the looped window. The standard 1/√L scaling (used in some depth-scaled models) is not enough and destabilizes multi-pass recurrence.

Our baseline has `attn_scale` and `mlp_scale` as **learned** `nn.Parameter(torch.ones(dim))` — initialized to 1.0. With NL=2 (3 total passes through layers 4–5), the correct init under this theory would be ~0.333 (= 1/3).

## Why our baseline might be overcooking the residuals

At NL=2 (loop passes 0, 1, 2), the looped layers 4–5 each contribute:
```
x += attn_scale * attn(norm(x))   ← 3 times for layer 4
x += mlp_scale * mlp(norm(x))     ← 3 times for layer 4
```

With `attn_scale = mlp_scale = 1.0` at init, the layer 4 residual is added 3× at full strength before the optimizer learns to soften it. This is fine if the optimizer can compensate quickly, but at our budget (~5000 steps, aggressive LR schedule), the first few hundred steps may be noisy. A 1/3 init gives the optimizer a better starting point.

## Implementation

Two options:

**Option A — pure init change (no param change)**

```python
# After creating looped block parameters:
if layer_idx in loop_layer_indices:
    with torch.no_grad():
        self.blocks[layer_idx].attn_scale.fill_(1.0 / num_loop_passes)
        self.blocks[layer_idx].mlp_scale.fill_(1.0 / num_loop_passes)
```

`num_loop_passes` = `NUM_LOOPS + 1` = 3 at NL=2.

This is hyperparam-only (no new params, no architectural change). Env var: `LOOP_SCALE_INIT=recip` (default: current 1.0 init).

**Option B — freeze the scale (no learning)**

Instead of learned `attn_scale`/`mlp_scale`, fix them to `1/L` for looped layers. Removes 2 × `model_dim` learned params per looped layer. Probably too aggressive — the learned gate does provide useful per-dim adaptation.

**Recommend Option A** — it's an init-only change that respects the existing learned gating.

## Interaction with current baseline

The `attn_scale` and `mlp_scale` are already learned in both looped and non-looped layers. This change only affects the **initialization** of those parameters in the looped layers. Non-looped layers (0–3, 6–10) keep `init = 1.0` as before.

No interaction with `RECUR_ALPHA_ENABLED`, `LOOP_ITER_EMBEDS`, or `MLP_ONLY_FROM` — all orthogonal.

## Screen plan

This is a hyperparam-only change (init values). Can be stacked onto any other loop-variant spec at zero extra cost. Recommend:

- Include as an arm in the iteration-embeddings screen (045A?) or MLP-only-loop screen
- Do NOT run as a solo spec — too subtle a change to justify a dedicated $4-6 screen
- If another loop spec runs, add a `+LOOP_SCALE_INIT=recip` arm as a parallel seed

## Risks / open questions

- **Does it matter at our step budget?** The optimizer may correct the 1.0 init within 100–200 steps, in which case 1/L init only helps the very early training (steps 0–200). With a ~5000 step budget and a cosine schedule, early steps may matter more than they look.
- **NL-dependent init**: if NL changes between runs (e.g., NL=2 vs NL=3), the right 1/L value changes. Need to tie init to `NUM_LOOPS` env var at launch time — easy to parameterize.
- **Non-looped layers**: do NOT apply 1/L init to non-looped layers — they are used exactly once per forward pass and the standard init is correct.

## If this works

Stackable with everything. A +0.001 or better result from this alone would justify locking it in as a permanent baseline change for all future specs. If it's noise (Δ < 0.0005), deprecate and move on — no code debt left behind since it's just an init value.
