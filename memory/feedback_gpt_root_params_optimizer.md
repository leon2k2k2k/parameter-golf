---
name: GPT-root parameters must be added to optimizer explicitly
description: Any nn.Parameter on the GPT root (not in self.blocks, not in the 4 bank tensors) is silently missing from all optimizer groups unless added by hand in Optimizers.__init__
type: feedback
---

`Optimizers.__init__` only sweeps `base_model.blocks.named_parameters()` for scalar params, and explicitly lists the 4 bank tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) for Muon. Any GPT-root `nn.Parameter` that doesn't fall into those two categories gets NO optimizer updates — it sits at init forever, silently.

**Why:** Discovered 2026-04-26 when `loop_iter_embeds` (Lever A, spec 045) was found to never have been trained. It was zero-init and stayed zero through all of Arms A, AC, E, F. All those runs tested C alone, not A+C. Three weeks of runs tainted.

**How to apply:** Every time you add a new `nn.Parameter` to the GPT class (not inside a block, not a bank tensor), immediately add a corresponding explicit append to `scalar_params` (or `matrix_params`) in `Optimizers.__init__`, guarded by `getattr(..., None) is not None and p.requires_grad`. Follow the pattern used for `smear_gate`, `recur_alpha`, `loop_iter_embeds`, `loop_resid_mixes`. Don't assume it gets picked up automatically.

Existing safe additions as of `fc54262`:
- `loop_iter_embeds` → scalar_params
- `loop_resid_mixes` → scalar_params
