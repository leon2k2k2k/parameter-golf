# Depth Recurrence in Transformers — Tight Literature Review

Date: 2026-04-27
Scope: same-pass depth recurrence only. Temporal recurrence (Block-Recurrent, Transformer-XL, RetNet, RWKV, Mamba, etc.) is excluded by design — see prior review (`recurrent-layer-attn-mlp-distribution-2026-04-27.md`) for the temporal-cell story.

## 0. Definition (so this is not conflated with temporal recurrence)

**Depth recurrence** in this review = a transformer that, in a *single forward pass on a single input sequence*, applies the same set of layer weights more than once to the same hidden states, producing iteratively refined representations. The recurrence is over *processing depth* (number of weight applications), not over *sequence positions*. Concretely: if `f` is a tied layer (or band of tied layers) with weights `θ`, the forward pass computes `h_{k+1} = f_θ(h_k)` for `k = 0..K-1`, then continues into the rest of the network. Block-Recurrent Transformer is **not** included here — its recurrence is across sequence chunks (temporal). UT, ALBERT-with-iteration, Looped Transformer, MoR, Geiping, ITT, Relaxed Recursive **are** depth-recurrent.

## 1. Per-paper, chronological

### 1.1 Universal Transformer — Dehghani et al., 2018 (arXiv:1807.03819)
- **Topology:** whole-stack tied. One unique transformer block (self-attn + transition FFN); the same block is applied T times. No "untied prelude / coda" wrapping it.
- **K:** fixed T or adaptive via ACT halting. WMT'14 base used T = 6–8 (matched to the 6-layer Transformer base it was compared against). Adaptive halting added on bAbI / LAMBADA; per-position ponder time learned.
- **Dims:** parameter-matched to vanilla Transformer base — `d_model=512`, FFN `d_ff=2048` (4×), 8 heads. Did **not** ablate dims of the recurrent body.
- **Per-pass differentiation:** identical weights every pass. The single per-pass signal is a (timestep, position) embedding added each iteration.
- **Headline:** +0.9 BLEU over vanilla Transformer base on WMT'14 En-De at matched params; SOTA on LAMBADA and bAbI at the time. (Source: arxiv.org/abs/1807.03819, openreview.net/pdf/6ee41939003eaa38439a2607d081864b4ba5fea4.pdf.)

### 1.2 ALBERT — Lan et al., 2019 (arXiv:1909.11942)
- **Topology:** whole-stack tied for parameter sharing (12 BERT layers all share one block). Each "layer" fires once at full depth — but the per-iteration view is identical to applying one block 12 times. So this is depth-recurrent in our sense.
- **K:** fixed at the depth of the equivalent BERT (12 for ALBERT-base, 24 for large). No K sweep.
- **Dims:** identical to BERT base — `d_model=768`, `d_ff=3072` (4×), 12 heads. The interesting move is factorized embeddings (E=128 vs E=768), not body dim.
- **Per-pass differentiation:** none.
- **Ablation that matters most for us — Table 4 (cross-layer sharing):** sharing only attention, only FFN, or all. *"Most of the performance drop … comes from sharing the FFN-layer parameters, while sharing the attention parameters results in no drop when E=128."* This is the cleanest "FFN-tying-hurts, attention-tying-is-free" data point in the literature. (Source: ar5iv.labs.arxiv.org/html/1909.11942.)

### 1.3 Looped Transformer (theory line) — Giannou et al., 2023 (arXiv:2301.13196) and Yang et al., 2023 (arXiv:2311.12424)
- **Topology:** whole-stack tied. A small transformer (13 layers in the programmable-computer paper) repeated in a loop. Yang et al. is the empirical follow-up showing looped transformers learn ICL function classes with fewer params.
- **K:** Yang et al. report results at K=12, 16, 30; the looped model with K iterations of a 1- to 2-layer body matches a 12-layer untied transformer on linear regression / sparse linear / decision-tree ICL.
- **Dims:** small ICL setting, `d_model` typically 256, FFN 4×, no ablation on body dim.
- **Per-pass differentiation:** none for Giannou; identical loop.
- **Headline:** matched performance with ~10% the parameters in their ICL function-class regime. Not language modeling.

### 1.4 Sparse Universal Transformer — Tan et al., 2023 (arXiv:2310.07096)
- **Topology:** whole-stack tied — **one shared block** with MoE attention (MoMHA, 24 attn experts) and MoE FFN, repeated for every layer. Stick-breaking ACT halt.
- **K:** evaluated 6, 8, 12 iterations. 6 was best on WMT14, 8 on CFQ, 12 on Logical Inference — task-dependent.
- **Dims:** `d_model=512`, FFN realized as MoE with k=4 of 24 experts; effective FFN width per pass is sparse-large but per-token compute is moderate.
- **Per-pass differentiation:** routing to experts can pick different experts each pass, so per-pass effective subnetwork differs even though weights are tied.
- **Headline:** 29.2 BLEU at 66M params on WMT14; +length generalization on Logical Inference (97% out-of-distribution).

### 1.5 Relaxed Recursive Transformer — Bae et al., 2024 (arXiv:2410.20672, DeepMind)
- **Topology:** **band-recurrence is the strongly recommended variant.** Take a pretrained Gemma/Pythia, pick a contiguous band of layers, tie + repeat that band. Layers outside the band can also be discarded vs. kept as untied prelude/coda — the paper tests both. The "Cycle"/"Sequence" sharing strategies and a Middle variant (unique first+last, share middle) are introduced here.
- **K:** repetition counts 2, 3, 4 of the recursive block; 3 most often optimal at the scales tested.
- **Dims:** inherited from the base model — Gemma-2B / Pythia: full `d_model`, full SwiGLU 8/3× FFN, full head count. They restore lost capacity via per-iteration LoRA rather than by adjusting body dim.
- **Per-pass differentiation:** **layer-wise LoRA per iteration.** This is the key relaxation: the same base weights, but a different small LoRA add-on for each pass.
- **Headline:** recursive Gemma-2B-rec recovers most of dense-2B accuracy at ~half unique-param count, with LoRA adapters closing the rest of the gap.

### 1.6 Geiping et al., recurrent-depth language model — 2025 (arXiv:2502.05171)
- **Topology:** **explicit prelude / recurrent core / coda — clean band recurrence.** Triplet `(l_P, l_R, l_C) = (2, 4, 2)`. The 4-layer core is repeated; the 2 prelude and 2 coda layers fire once.
- **K:** trained with a Poisson-Lognormal sampling around mean `r̄ = 32`. Inference tested at r ∈ {1, 4, 8, 16, 32, 64, 128}. Test-time more iterations → monotone gains on reasoning until ~32–64.
- **Dims:** `d_model = 5280`, 55 heads of dim 96, MLP inner 17920 (~3.4×). 3.5B active params; same dim across prelude/core/coda.
- **Per-pass differentiation:** none — pure tying. They lean on the recurrence-as-test-time-compute story.
- **Ablation on band partition:** **none reported.** They picked (2,4,2), did not vary it. Section 4.3 only ablates norms/init at scale.
- **Headline:** 3.5B model at high recurrence matches a ~50B dense baseline on GSM8K / reasoning at deploy time.

### 1.7 Inner Thinking Transformer (ITT) — Chen et al., 2025 (arXiv:2502.13842, ACL 2025)
- **Topology:** **interleaved band, not contiguous.** *"We replace every other layer of original model with a Loop or ITT layer."* So the ITT (looping) layers alternate with standard untied layers — this is the closest setup in the literature to "loop a *subset* of layers."
- **K:** thinking steps tested 2×, 3×, 4×; 4× best, diminishing returns past 4.
- **Dims:** LLaMA-2-style. Hidden `d_model = 1024 / 1536 / 2048` for the 162M / 230M / 466M scales, 4× FFN ratio (Llama-2 default), no shrink.
- **Per-pass differentiation:** Residual Thinking Connection (residual carries state between iterations) + Thinking-Step Position Encoding (per-iteration embedding). Adaptive Token Routing decides per token whether to take an extra iteration.
- **Headline:** ITT-162M reaches 96.5% of a 466M Transformer baseline on 11 benchmarks. Strongest "looping is a real param-efficient lever" empirical result on small-scale LM.

### 1.8 Mixture-of-Recursions (MoR) — Bae et al., 2025 (arXiv:2507.10524, NeurIPS 2025)
- **Topology:** four sharing strategies tested side-by-side on the **same** stack: Cycle, Sequence, Middle-Cycle, Middle-Sequence. **Middle-Cycle = first + last layers are unique, middle layers tied + cycled. This is the closest published equivalent of band-recurrence for LM.** Middle-Cycle wins.
- **K (recursion depth N_r):** 2, 3, 4. N_r = 3 most often the best Pareto point.
- **Dims:** Llama / SmolLM-style — full d_model, full SwiGLU FFN, full head count, no body shrink.
- **Per-pass differentiation:** none architecturally; differentiation comes from token-level routing — a router decides per-token how many of the N_r recursions a token participates in. Some tokens loop 1×, some 3×.
- **Ablation that matters for us:** Figure 4(b) shows Middle-Cycle > Cycle ≈ Sequence > Middle-Sequence at both 135M and 360M scales over 10B tokens. So *position matters, and "loop the middle, keep first+last unique" wins*.
- **Headline:** at 16.5e18 FLOPs, MoR (N_r=2, expert-choice routing, KV cache) NLL 2.7511 vs vanilla 2.7824, with ~half unique params; 2× inference throughput.

### 1.9 Improving Recursive Transformers with Mixture of LoRAs ("ModernALBERT") — Dec 2025 (arXiv:2512.12880)
- **Topology:** whole-stack tied (ALBERT-style). Per-pass adapters are routed: a Mixture-of-LoRAs inside the shared FFN.
- **K:** matched to the depth of the dense baseline.
- **Dims:** inherited; the paper's lever is the LoRA expert pool, not body dim.
- **Per-pass differentiation:** token-conditional LoRA experts inside the tied FFN.
- **Headline:** "parameter sharing in recursive transformers reduces model size but collapses layer-wise expressivity" — confirmed; their MoLoRA fix recovers dense-baseline quality.

### 1.10 Two-Scale Latent Dynamics for Recurrent-Depth Transformers — Pappone et al., 2025 (arXiv:2509.23314)
- **Topology:** analysis paper, not new architecture. They study Geiping-style recurrent-depth transformers and a GPT-2-like model with three different recurrent regions.
- **What they observe:** within a looped block, updates shrink and become orthogonal across iterations; across blocks the state drifts more globally. This is the cleanest evidence that depth-recurrence iterations do *fine refinement*, not gross transformation — supporting the "recurrent body is doing different work than the surrounding stack" view.
- They propose a second-order halt criterion (consecutive update angle plateaus → stop).

### 1.11 Retrofitted Recurrence — Nov 2025 (arXiv:2511.07384)
- **Topology:** explicit band-recurrence on pretrained models. **(4, 8, 4)** triplet on TinyLlama-1.1B (4 prelude / 8 recurrent / 4 coda) with explicit ablation on which layers to put where: *"selecting the early layers for the prelude and later layers for the recurrent block and coda performs best."* Closest direct ablation on band position published.
- **K:** Poisson-Lognormal sampling around mean = 32 (matching Geiping training setup). Tested r=8 to r=64 at deploy.
- **Dims:** inherited from TinyLlama / OLMo / Llama-3.2 — full base dims; no body shrink.
- **Per-pass differentiation:** none beyond initialization from pretrained weights and position-of-iteration handling.
- **Headline:** retrofitted (4,8,4) TinyLlama at r=32 → 52.0% GSM8K vs 46.2% post-trained baseline at matched compute.

### 1.12 AdaPonderLM — 2026 (arXiv:2603.01914)
- **Topology:** depth-recurrent LM with token-wise adaptive halt and a learned gate (a "ponder" gate). Closer to UT-with-ACT generalized to LM.
- **K:** mean ponder time ~3–5 per token at trained budgets. Adaptive.
- **Dims:** standard Llama-style.
- **Headline:** matches non-recurrent baseline val_loss with fewer effective forward passes per token. Mostly a halt-mechanism contribution.

### 1.13 Pre-LN / Post-LN inside tied stacks
The literature here is thin and not directly ablated in any of the depth-recurrent papers. UT used the original Post-LN recipe (and noted instability that they patched with extra residual care); ALBERT followed BERT's Post-LN; Geiping (2502.05171) explicitly uses sandwich-style Pre-LN + extra normalization in the recurrent core to keep r=32 iterations stable, and Section 4.3 reports that Post-LN diverged at scale in the recurrent regime. Two-Scale Latent Dynamics (2509.23314) confirms iterations get smaller and more orthogonal under Pre-LN — i.e. Pre-LN keeps the loop well-conditioned. **Consensus is qualitative but unanimous: Pre-LN (often with extra core normalization) is the only thing that has been shown to train stably at high K in depth-recurrent LMs.**

## 2. Cross-paper synthesis

### 2.1 Consensus on K (number of passes)
- **Whole-stack tied LMs at small/medium scale (UT, SUT, ALBERT, ITT, MoR):** sweet spot is **K = 2 to 4**. Beyond 4, returns are flat or negative for language modeling on val_loss. ITT explicitly: 4 best, diminishing past 4. MoR: N_r = 3 the best Pareto point. UT: 6–8 was matched-to-baseline, no clear win past that.
- **High-K test-time-compute regime (Geiping, Retrofitted Recurrence, AdaPonderLM):** they train with mean K ≈ 32 and report monotone gains on *reasoning* tasks (GSM8K, MATH) up to K = 32–64. But on plain LM val_loss, the gain saturates much earlier — high-K is a deploy-time reasoning trick, not a pretraining-loss trick.
- **Implication for our 50M LM regime:** literature points to K ∈ {2, 3, 4}. We are using NL=2 (3 passes total) which is the median of this range. Going to NL=3 (4 passes) is consistent with ITT's optimum.

### 2.2 Consensus on dims of the recurrent body
**Striking consensus: keep the recurrent body at full base dimensions.** Every paper above either inherits the base model dims (UT, ALBERT, SUT, MoR, Geiping, Relaxed Recursive, ITT, Retrofitted) or, when they want more capacity, adds it via LoRA / experts inside the tied block (Relaxed Recursive, MoLoRA). **Not one of these papers ablates a narrowed `d_ff` or `d_model` inside the recurrent body and reports the result.** The closest is Block-Recurrent Transformer's MLP-skip ablation — but that's *temporal*, so under our scope it doesn't count.

**This means:** "shrink the looped body" is genuinely under-tested in the depth-recurrent literature. Our prior review (`recurrent-layer-attn-mlp-distribution-2026-04-27.md`) leaned on Block-Recurrent for this — that's a temporal-cell result, and applying it to depth recurrence requires an explicit caveat.

### 2.3 Where papers disagree
- **Differentiation per pass.** Pure tying (UT, ALBERT, Geiping, Looped) vs. relaxed tying (Relaxed Recursive's LoRA, MoLoRA's experts, ITT's per-step encoding, MoR's per-token routing). The relaxed-tying papers are unanimous that pure tying loses capacity vs. matched dense, and the fix is *adding* small per-pass parameters — never *shrinking* the body. The pure-tying papers either don't make a matched-dense comparison (UT didn't, beyond +0.9 BLEU vs same-param baseline) or accept the gap as the price of param efficiency (ALBERT).
- **Whole-stack vs. band recurrence (the central question).** UT / SUT / ALBERT / Looped / Geiping (originally) treat depth recurrence as a whole-stack lever; Relaxed Recursive, MoR, ITT, Retrofitted Recurrence treat it as a band lever inside an otherwise normal stack. The newer (2024–2025) consensus is shifting toward band recurrence — three of the four most recent papers (MoR, ITT, Retrofitted) explicitly use band+wrapper.
- **Halt mechanism.** Fixed K (Geiping, Looped, ALBERT) vs. adaptive (UT/ACT, SUT, ITT, AdaPonderLM, MoR per-token routing). At pretraining scale on val_loss, fixed K with `K=2..4` consistently matches or beats adaptive.

### 2.4 Has anyone done band recurrence cleanly?
Yes — and recently. The clean cases:

| Paper | Band format | Wrapper untied? | Position ablated? |
|---|---|---|---|
| **MoR (2507.10524)** | Middle-Cycle: tied middle, unique first + last | Yes (1 unique layer at each end) | Yes — Middle-Cycle > Cycle ≈ Sequence > Middle-Sequence |
| **Geiping (2502.05171)** | (l_P, l_R, l_C) = (2, 4, 2) | Yes (2 prelude + 2 coda) | **No** — they fixed (2,4,2) |
| **Retrofitted Recurrence (2511.07384)** | (4, 8, 4) on TinyLlama | Yes | **Yes** — "early layers for prelude, later layers for recurrent + coda performs best" |
| **ITT (2502.13842)** | Every-other layer is a tied loop layer | Yes (alternating untied) | Implicit — they set the alternation pattern, did not ablate band width directly |
| **Relaxed Recursive (2410.20672)** | Contiguous middle band on pretrained Gemma/Pythia | Optional | Partially |

**Where they directly answer "does band position matter?":**
- **MoR Figure 4(b):** put unique layers at the *ends*, not the middle. Middle-Cycle (loop the middle) beats Cycle (loop everything).
- **Retrofitted Recurrence:** put early layers in the *prelude*, later layers in the *recurrent + coda*. Implies the input-side of the network benefits least from looping; the deeper layers benefit most.

**Where they do NOT answer it:**
- Width of the band. Nobody published "loop 2 layers vs 4 vs 8 in a fixed-size stack." Geiping fixed (2,4,2). MoR sweeps recursion depth N_r but holds the band size fixed per scale. Retrofitted fixed (4,8,4) for the main results.
- Fraction of total stack in the band. Geiping is 4/8 = 50%, Retrofitted is 8/16 = 50%, ITT is alternating ≈ 50%. **The depth-recurrent literature converges around ~50% of the stack being the looped band, but nobody explicitly justifies it with a sweep.**

## 3. Verdict for our regime

Our regime: 12-layer untied transformer, ~50M params, language-modeling pretraining objective, layers {3, 4, 5} tied and run NL = 2 (3 passes). Band fraction = 3/12 = 25%, K = 3, position = early-middle.

What the literature says:

**1. K = 3 is in the supported range.** ITT 4× best, MoR N_r = 3 best, UT optimum 6–8 of a single layer (so per-token weight applications ~6–8) — our 3 sits at the lower end. Consistent. Going to NL=3 (K=4) is *also* in-range and matches ITT's optimum. NL=4 (K=5) starts to leave the supported range for whole-stack-tied LMs and is not recommended without a different halt / per-pass differentiation mechanism.

**2. Band position is *not* what the consensus picks.** MoR's winning Middle-Cycle puts unique layers at *both ends* and loops the middle. We loop {3, 4, 5} — early-middle. Retrofitted Recurrence specifically says "early layers for the prelude" — i.e., do NOT loop early layers; put them in the prelude. Our setup loops layers that should arguably be in the prelude. **This is a flag.** A sibling spec that loops {5, 6, 7} (true middle) or {6, 7, 8} (middle-late) would be more aligned with both MoR and Retrofitted Recurrence.

**3. Band fraction (25%) is below the published norm (~50%).** Nobody has shown 25% works as well as 50%, but nobody has shown it doesn't either. Could be a real difference; could be fine. **Genuinely untested at our scale.**

**4. Body dim should be full.** Every depth-recurrent LM paper keeps the looped body at full base dims. Shrinking the looped body's `d_ff` or `d_model` is not validated by the depth-recurrent literature; the closest (Block-Recurrent) is temporal and out of scope here. The earlier `recurrent-layer-attn-mlp-distribution` review's "shrink the MLP" recommendation needs the explicit caveat that *all the supporting evidence is from temporal-cell papers, not from any depth-recurrent ablation*.

**5. Per-pass differentiation is where the modern wins live.** Pure tying (our current setup) is the same path UT and ALBERT took, and the same path Relaxed Recursive / MoLoRA / ITT explicitly improved on by adding per-pass perturbations (LoRA, residual thinking connection, step encoding). If our looped body is bottlenecked, the literature's recommended fix is *per-iteration LoRA or per-iteration position-of-pass embedding*, not body shrink and not removing tying.

**6. Pre-LN / sandwich norm is non-negotiable at our K.** Geiping at r=32 needed extra normalization in the core; we are at K=3 so this is less urgent, but Post-LN inside the tied band has historically been the failure mode. Verify the recurrent layers are Pre-LN.

## 4. Open empirical questions the literature has NOT answered

For our setup specifically, every one of these is unstudied and a candidate for our own ablation:

1. **Band width.** Does 2-layer band tied beat 3-layer? 4-layer? Nobody has published this sweep at small-LM scale. (Geiping/Retrofitted both fixed band ~50% of stack and stopped.)
2. **Band position when fraction is small (≤25% of stack).** All published sweeps (MoR, Retrofitted) examined ~50% bands. With only 3/12 layers in the band, the optimal *position* could differ from MoR's middle-loop-with-unique-ends rule.
3. **K vs band width tradeoff at fixed compute.** If we have a budget of 6 extra layer-applications, is it better to loop a 2-layer band 3 extra times (K=3) or a 3-layer band 2 extra times (K=2)? Untested.
4. **Body shrink in a depth-recurrent (not temporal) loop.** All the "shrink the loop body" intuition comes from Block-Recurrent (temporal). Whether shrinking `d_ff` in a depth-recurrent band hurts or helps is genuinely open.
5. **Per-iteration position-of-pass embedding at small K.** UT used (timestep, position) embeddings at every iteration; ITT uses Thinking-Step Position Encoding. Whether this matters at K=3 vs K=32 is not sweeped.
6. **Asymmetric tying inside the band (attention tied, FFN untied — or vice versa).** ALBERT shows attention tying is approximately free and FFN tying hurts. No one has applied this to a *band* (rather than whole-stack) — i.e. tie attention across passes inside the band but give each pass its own FFN.
7. **Interaction with Pre-LN scale init / sandwich norm at small K.** Geiping needed extra norm at K=32; whether the same recipe helps or hurts at K=3 has not been tested.
8. **Band recurrence + EMA / phased-TTT** — the levers we already have on the leaderboard. Cross-talk with depth recurrence is not in any paper above.

## 5. Sources

- Universal Transformer — arxiv.org/abs/1807.03819, openreview.net/pdf/6ee41939003eaa38439a2607d081864b4ba5fea4.pdf
- ALBERT — ar5iv.labs.arxiv.org/html/1909.11942
- Looped Transformer (theory) — arxiv.org/abs/2301.13196
- Looped Transformer (ICL empirics, Yang et al.) — arxiv.org/abs/2311.12424
- Sparse Universal Transformer — arxiv.org/html/2310.07096
- Relaxed Recursive Transformer — arxiv.org/abs/2410.20672
- Geiping recurrent-depth — arxiv.org/abs/2502.05171, arxiv.org/html/2502.05171v1
- Inner Thinking Transformer — arxiv.org/abs/2502.13842, arxiv.org/html/2502.13842v2
- Mixture-of-Recursions — arxiv.org/abs/2507.10524, arxiv.org/html/2507.10524v1
- Two-Scale Latent Dynamics — arxiv.org/abs/2509.23314
- Retrofitted Recurrence — arxiv.org/abs/2511.07384, arxiv.org/html/2511.07384v1
- AdaPonderLM — arxiv.org/abs/2603.01914
- Improving Recursive Transformers w/ MoL ("ModernALBERT") — arxiv.org/abs/2512.12880
