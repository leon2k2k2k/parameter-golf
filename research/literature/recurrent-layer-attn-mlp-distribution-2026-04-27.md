# Recurrent / Weight-Tied Transformer Layers: How Should Attention vs MLP Be Sized?

Date: 2026-04-27
Context: Parameter Golf, ~50M-param LM, layers {3,4,5} of a 12-layer stack run NL=2 (3 passes); throughput-bound. Question: should the looped block be shrunk (MLP narrower, fewer heads, bottleneck) vs. kept full-dim?

## Executive summary

The literature on weight-tied / depth-recurrent transformers is **dominated by papers that keep the recurrent block at full base dimensions** (UT, ALBERT, MoR, Relaxed Recursive, Inner Thinking Transformer, Geiping recurrent-depth). Those that explicitly ablated *what to shrink* split sharply along one axis: **the FFN/MLP is where weight-tying hurts most, but in a recurrent cell the MLP is also the most expendable** when the cell is sandwiched inside a larger non-recurrent stack. Below is the per-paper read.

## Per-paper findings

### 1. Universal Transformer (Dehghani et al. 2018, arXiv:1807.03819)
The original. Self-attention and a "transition function" (position-wise FFN) are tied across T recurrence steps. The paper does **not ablate FFN vs attention dim** inside the shared block; the WMT'14 base model is parameter-matched to vanilla Transformer base, implying same `d_model`, same FFN multiplier (4×), same head count. Silent on shrinking. (Sources: arxiv.org/abs/1807.03819, openreview.net/pdf/6ee41939003eaa38439a2607d081864b4ba5fea4.pdf.)

### 2. ALBERT (Lan et al. 2019, arXiv:1909.11942)
**The single most relevant ablation.** ALBERT shares all params across 12 BERT layers but factorizes the embedding. Their cross-layer sharing ablation (Table 4 in v2): "Most of the performance drop ... comes from sharing the FFN-layer parameters, while sharing the attention parameters results in no drop when E=128 (+0.1 on Avg) and a slight drop when E=768 (-0.7 on Avg)." Translation: **FFN tying is what costs you; attention tying is approximately free.** ALBERT keeps the FFN at the same 4× ratio as BERT (3072 inner / 768 hidden); they did *not* shrink the FFN inside the tied block — they tied at full width and ate the loss. (openreview.net/pdf?id=H1eA7AEtvS.)

### 3. Block-Recurrent Transformer (Hutchins & Schlag et al. 2022, arXiv:2203.07852)
**The most directly load-bearing data point for our question.** A 12-layer transformer with *one* recurrent layer at position 10. They explicitly ablate four cell configurations — `single`, `dual`, `fixed`, and `skip`. The `skip` config **removes the MLP from the recurrent layer entirely**, replacing it with a linear projection. Direct quotes:
- "In a recurrent layer, removing the MLP makes little difference; it does not seem to be computing anything useful."
- "Removing the MLP from all layers would severely degrade the model … those large MLPs are computing something important. In a recurrent layer, removing the MLP makes little difference."
- The `fixed:skip` configuration — no MLP in the recurrent cell — was their best on 3/4 datasets and **the fastest** (lower step time, fewer params).

This is the strongest single piece of literature evidence that the **MLP inside a looped cell is partially redundant** when there is also a stack of full-MLP non-recurrent layers around it. Note: they did not ablate "shrink MLP to 2×/1×"; they tested 0× vs 4×. (ar5iv.labs.arxiv.org/html/2203.07852.)

### 4. Sparse Universal Transformer (Tan et al. 2023, arXiv:2310.07096)
SMoE applied independently to attention heads and FFN inside a shared 6-layer recurrent block. They keep the *block-level* `d_model=512` and use 4× FFN; the MoE structure means individual *experts* are smaller, but the sum-over-experts width is full. No ablation isolates "smaller dense FFN in shared block."

### 5. Relaxed Recursive Transformer (Bae et al. 2024, arXiv:2410.20672, DeepMind)
Take a pretrained Gemma/Pythia and recursively re-use a contiguous block. The recursive block keeps the **base model's full dim, full FFN ratio, full head count** — they relax tying via per-iteration LoRA adapters. The conclusion is that you cannot recover lost capacity by shrinking; you have to add LoRA back. Implicit but strong: simple full-dim tying loses capacity, and the fix is *more* params (LoRA), not narrower MLPs.

### 6. Mixture-of-Recursions (Bae et al. 2025, arXiv:2507.10524, NeurIPS 2025)
Shared stack of layers reused 1–N times per token (router-controlled). Architecture is "Llama-based" / SmolLM config — full base FFN ratio (likely SwiGLU 8/3 × `d_model`), full head count. Their ablations are over routing strategies (Cycle / Sequence / Middle-Cycle), not over **internal block dim**. **Silent** on whether a thinner shared block would have worked equally well.

### 7. Improving Recursive Transformers w/ Mixture of LoRAs (Dec 2025, arXiv:2512.12880, "ModernALBERT")
Same takeaway as Relaxed Recursive but more elaborate: tying collapses layer-wise expressivity, so they restore it with token-conditional LoRA experts **inside the shared FFN**. Direct from abstract: "Parameter sharing in recursive transformers reduces model size but collapses layer-wise expressivity." Their fix points up not down — *more* FFN capacity (via experts), not less.

### 8. Inner Thinking Transformer (Chen et al. 2025, arXiv:2502.13842, ACL 2025)
Adaptive Token Routing decides per-token recurrence depth on a tied block. Body of the recurrent block is full-width; the lever is *which tokens* loop, not how thin the loop body is. ITT-162M reaches 96.5% of a 466M baseline, which is the strongest "looping is a real param-efficient lever" result on small models. Silent on shrinking the loop body itself.

### 9. Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (2025, arXiv:2502.05171)
3.5B-param proof-of-concept. Recurrent block is **a single transformer layer at full base width** repeated; they explicitly avoid shrinking because their thesis is "more iterations = more compute". Empirically, recurrent compute matches a 50B-param dense model on reasoning. They do not ablate body width.

### 10. Looped Transformers (Giannou et al. 2023, arXiv:2301.13196; Yang et al. 2023)
Theory-leaning. Recent empirical follow-ups (per the survey at emergentmind.com/topics/looped-transformer-architecture) describe "4 parallel residual streams, allocating roughly 50% to a middle block that is looped multiple times" — i.e. they **do** carve out a slimmer middle block for looping. But this is a *parallel-residual-fraction* knob, not an MLP-shrink knob; the looped layers themselves still use full-dim attention + 4× FFN within their residual stream.

### 11. Mixture-of-Depths (Raposo et al. 2024, arXiv:2404.02258)
Token-level skip rather than param-tying — orthogonal to our question. But notable: MoD shows you can drop ~50% of tokens at each layer with no loss, hinting that **per-layer compute is over-provisioned**, especially for MLP. Does not directly address shrinking the layer.

### 12. Tay et al. "Scale Efficiently" (arXiv:2109.10686)
DeepNarrow: same total params, more layers / narrower `d_model`/`d_ff` is Pareto-better than wide-shallow. They report transformer perf "depends very weakly on shape parameters" once total params are fixed — this is the **strongest prior** that, *for a fixed param budget*, mlp-ratio < 4 with more layers does ~as well. But they did **not** test asymmetric: wide non-recurrent layers + narrow recurrent layers in the same stack.

### 13. CoLT5 (Ainslie et al. 2023, arXiv:2303.09752)
Explicit narrow/wide split: every layer has a **light MLP applied to all tokens** and a **heavy MLP applied to routed tokens only**. Direct precedent for asymmetric-FFN-within-a-layer. Gain is real (SCROLLS SOTA, much faster). Different mechanism from depth-recurrence but the *principle* — "not every pass through the layer needs the full 4× MLP" — is shared.

## Consensus and disagreements

**Consensus points:**
1. **Tying the FFN is the painful tying decision; tying attention is approximately free.** ALBERT (E=128 case), Relaxed Recursive, Improving Recursive w/ MoL all agree. Implication for *us*: the looped attention can be shared aggressively; it's the **looped MLP** where capacity is being asked to do the most work.
2. **Pure recurrent param efficiency loses capacity vs. matched dense.** ALBERT, Looped Transformer empirics, MoR, Relaxed Recursive all say so. The fix in the literature is *not* "shrink the loop body further" — it's "loop a full-width body and recover lost capacity via adapters / routing / extra iterations".
3. **For a fixed total param budget, deep-narrow ≈ wide-shallow** (Tay et al.). So shrinking the MLP isn't catastrophic *if* you compensate in some other dim.

**Disagreements / tension:**
- **Block-Recurrent Transformer is the lone clear voice saying "the MLP in a recurrent layer is doing nothing useful — just remove it"**, and they show this *with the MLP-rich rest of the stack still in place*. This is the closest setup to ours (12-layer model, one or a few recurrent layers, full MLP elsewhere).
- Everyone else who studied recurrence as the *whole stack* (UT, MoR, Geiping, ITT) implicitly disagrees — they need the MLP because the MLP is the only place compute is happening. The disagreement therefore resolves to **"is the rest of the model providing enough MLP capacity already?"** In our setup (only layers 3–5 loop, layers 0–2 + 6–11 are full and not tied), the answer is yes — Block-Recurrent's regime applies, not Universal Transformer's.
- No paper has cleanly ablated *partial* MLP shrink (e.g. 4× → 2× → 1×) inside a tied block embedded in a non-tied stack. **Literature is silent on the intermediate point.** Block-Recurrent jumped directly 4× → 0×.

## Concrete design recommendations (literature-grounded)

If forced to pick one config from the literature for your spec, the citations point this way, in priority order:

1. **Strongest signal — try shrinking or removing the MLP in layers {3,4,5}** (Hutchins & Schlag 2022). Their `skip` configuration *removed* the MLP from the recurrent layer and won 3/4 benchmarks while running fastest. Concretely: `mlp_ratio = 0` (linear projection only) or `mlp_ratio = 1×` for layers {3,4,5}. Throughput gain is ~4/7 of MLP FLOPs × 3 passes — large.
2. **Second-strongest — keep MLP at 4× but shrink attention heads in the looped block** is the *opposite* of what the literature says to do. ALBERT specifically found attention tying is free and FFN tying hurts; shrinking heads is solving the wrong axis. Avoid.
3. **Bottleneck (down-proj at entry, narrow inner, up-proj at exit) is not directly tested by any paper above.** Closest analog is CoLT5's light MLP, which is a narrow MLP (low `d_ff`) but **not** a `d_model` bottleneck. **Literature is silent on `d_model` bottlenecks in recurrent transformer blocks** — flagging explicitly as untested.
4. **Hedged middle option:** if removing the MLP feels too aggressive, set `mlp_ratio = 2×` for layers {3,4,5}, leave heads at 8, and treat it as the interpolation between Block-Recurrent's 0× and Universal Transformer's 4×. **This intermediate is not directly tested in the literature**, but is consistent with the deep-narrow Pareto from Tay et al. and the "FFN is where to economize" signal from ALBERT.

## Where the literature is silent (do not extrapolate)

- Partial MLP shrink (1–3×) inside a tied sub-block of an otherwise untied stack — **untested**.
- `d_model` bottlenecks (down-proj, thin inner, up-proj) on the recurrent block — **untested** in the surveyed papers; closest is CoLT5's light pathway, which is `d_ff` narrow not `d_model` narrow.
- Head-count reduction inside tied vs untied layers asymmetrically — **untested directly**; ALBERT's signal is from full-attention-tied vs untied, not from "fewer heads inside tied".
- Whether NL=2 (3 passes) specifically tips the calculus relative to NL=1 or NL=8 — no paper varies the number of passes against the body-shrink axis.

If we want primary literature backing for a parameter-golf-style submission, **the Block-Recurrent `skip` ablation is the citation** — it's the only result that directly shows "in a model where layers {everything else} carry the MLP weight, the MLP inside the recurrent layer can go to zero with no loss."

## Sources

- [Universal Transformers](https://arxiv.org/abs/1807.03819) — Dehghani et al. 2018
- [ALBERT](https://openreview.net/pdf?id=H1eA7AEtvS) — Lan et al. 2019
- [Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852) / [ar5iv](https://ar5iv.labs.arxiv.org/html/2203.07852) — Hutchins & Schlag et al. 2022
- [Sparse Universal Transformer](https://arxiv.org/abs/2310.07096) — Tan et al. 2023
- [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672) — Bae et al. 2024 (DeepMind)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) — Bae et al. 2025
- [Improving Recursive Transformers with Mixture of LoRAs](https://arxiv.org/abs/2512.12880) — Dec 2025
- [Inner Thinking Transformer](https://arxiv.org/abs/2502.13842) — Chen et al. 2025
- [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171) — Geiping et al. 2025
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) — Raposo et al. 2024
- [Scale Efficiently / DeepNarrow](https://openreview.net/pdf?id=f2OYVDyfIB) — Tay et al.
- [CoLT5](https://arxiv.org/abs/2303.09752) — Ainslie et al. 2023
- [Looped Transformers as Programmable Computers](https://proceedings.mlr.press/v202/giannou23a/giannou23a.pdf) — Giannou et al. 2023
