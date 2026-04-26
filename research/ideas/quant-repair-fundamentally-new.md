# Quant repair — fundamentally new ideas (post-046 family)

**Created:** 2026-04-27
**Status:** brainstormed, none specced yet
**Context:** After 046 family closed (see `quant-repair.md`), only path to more BPB
on the quant side is fundamentally new techniques, not variants of SDClip/LQER/etc.

## Why these are "new" vs prior 046 work

The 046 family explored variants of techniques already in the stack:
- SDClip tightening (proven, byte-blocked)
- LQER knob sweeps (null)
- Calibration variations (null)
- Post-quant fitting (hurts at artifact-time)
- AR self-gen calib (null at artifact-time)

These ideas are **different paradigms** — different storage formats, different
repair mechanisms, or different deploy-vs-artifact split.

---

## A. Deploy-time quant repair using eval headroom (HIGHEST EV)

**The paradigm shift**: artifact-time → deploy-time repair.

PR #1797 used only 423-495s of the 600s eval cap. Every submission since has
**100-180s of unused leaderboard compute**. This time can run a quant repair
step at deploy time, bypassing the 16MB cap entirely.

### Concrete mechanism

```python
# In eval pipeline (after deserialize):
calib_data = ar_self_gen(model, n_batches=16, seq_len=512)  # ~30s
fit_passthrough_params(eval_model, calib_data, iters=5)  # ~30s using 046E logic
# Then standard TTT + eval
```

Combined cost: ~60s. Fits in 100-180s headroom.

### What we already have

- 046E (`fit_passthrough_params_to_match_base`): full code in train_gpt.py
  commit 381baf2. Failed as artifact-time lever (no propagation), but the
  fit logic itself is sound.
- 046F (`ARSelfGenCalibLoader`): full code, generates AR samples without
  val data leak. Already legal for calibration use.

### What's missing

- Wire both into the eval path (vs current train_and_eval path)
- Use the unquantized loaded weights as "teacher" instead of needing
  base_model in scope (could keep a pre-quant snapshot in memory)
- Verify rules legality — unclear if updating quant artifacts at deploy
  time counts as "training" (TTT clearly is allowed; this is similar in spirit)

### Cost / EV

- Code: ~50-100 lines (mostly wiring)
- Test cost: ~$5
- EV: -0.001 to -0.005 BPB at deploy time (literature for similar approaches)
- Risk: rules question + complexity

### Why this might actually work where 046E failed

- 046E fit at artifact-time → fitted values were in-memory only, didn't
  propagate to artifact → screen showed cost (+0.0004) but no shipping value
- DEPLOY-time fit → values are applied to the CURRENTLY-RUNNING model →
  directly impacts the eval being scored
- TTT works exactly this way and recovers ~-0.013 BPB

---

## B. NF4-style non-uniform quantization (NormalFloat 4-bit)

**The paradigm shift**: linear levels → distribution-fit levels.

Standard int6 has 64 evenly-spaced levels. NF4 (Dettmers et al., QLoRA 2023)
picks levels to minimize MSE for the actual weight distribution.

### Why for our stack

Muon-trained weights are approximately Gaussian (sub-Gaussian per the
SpinQuant null findings). NF4 was designed exactly for this case.

Same total bytes as int6 (64 codepoints encoded the same way), but each
weight maps to the closest non-uniform codepoint instead of the closest
linear bin. Better fit at zero byte cost.

### Cost / EV

- Code: ~30-50 lines (replace linear bin assignment with non-uniform table)
- Cost: ~$3-5 to test
- EV: -0.001 to -0.003 BPB (literature: NF4 vs int4 dramatic; int6 vs NF6 less dramatic but still real)
- Risk: low (same byte budget, just better-fitted levels)

### Implementation note

Need to recompute the optimal codepoints for OUR weight distribution.
Standard NF4 codepoints assume zero-mean unit-variance Gaussian; Muon
weights might have different statistics that warrant custom levels.

---

## C. Vector quantization with shared codebook

**The paradigm shift**: per-element scalar quant → vector quant with codebooks.

Group small chunks of weights (e.g., 16 elements at a time) into vectors,
quantize each vector to one of N codebook entries. **Codebook is shared
across all tensors** to amortize codebook overhead.

### Why VQ rather than scalar quant

For correlated weight elements (which exist in trained matrices), VQ can
be much more efficient per byte than scalar quant. Standard VQ overhead
makes it lose at LLM scale, but **shared codebooks change the math**.

### Cost / EV

- Code: ~80-150 lines (codebook fitting via k-means + lookup machinery)
- Cost: ~$5-10
- EV: -0.002 to -0.005 BPB at the right grain
- Risk: medium (codebook fitting is finicky, deserialize complexity)

### Why this might fail

LLM weights don't have strong cross-element correlation at small chunk
sizes (16 elements). Most VQ wins in literature come from larger chunks
(32-256) which need bigger codebooks. Our 16MB cap limits codebook size.

---

## D. Self-distillation during quantization

**The paradigm shift**: per-layer reconstruction MSE → multi-layer KL distillation.

GPTQ minimizes per-layer reconstruction error (one layer at a time, weight
output matching). Self-distillation: minimize KL divergence between full FP
forward output and full quant forward output **across multiple layers
simultaneously**.

### Why this might give more

GPTQ's per-layer objective doesn't account for error propagation across
layers. By the 11th block, accumulated error is much larger than any single
layer's reconstruction error. Multi-layer distillation directly targets
end-to-end output preservation.

### Cost / EV

- Code: ~150-300 lines (multi-layer forward pass during quant decision,
  back-prop or coordinate descent over rounding)
- Cost: multi-day code work
- EV: -0.002 to -0.005 BPB
- Risk: high (complex implementation, may overlap with what GPTQ already
  captures via Hessian)

### Why this might fail

GPTQ's Hessian already captures activation statistics, which is a proxy
for what distillation would capture. The marginal gain might be small.

---

## E. Tensor-train decomposition for embedding

**The paradigm shift**: dense storage → tensor-network factorization.

Express the 8192×512 token embedding as a chain of smaller tensors
`T1 @ T2 @ T3` where each Ti is much smaller. With proper decomposition
rank, can save 50-80% of embedding bytes.

### Why for embedding specifically

`tok_emb` is the largest single tensor (~3.7 MB after current quant).
Tensor-train is well-suited to lookup-table-style structure (which is
exactly what an embedding is — a lookup table).

### Cost / EV

- Code: ~100-200 lines (TT decomposition + custom forward through TT cores)
- Cost: ~$5 to validate
- EV: 1-2 MB potential byte savings (HUGE)
- Risk: high (changes model expressivity for embeddings; may damage val_bpb)

### Why this is a huge bet

If it works, it frees enormous byte budget — enough to ship 046G-tightest
plus add MORE LQER plus other improvements. If it doesn't, the embedding
expressivity damage could be worse than the byte budget unlocked.

---

## Ranking by EV / risk-adjusted

| # | Idea | Code | Test cost | Plausible win | Risk |
|---|---|---|---|---|---|
| **1** | **A. Deploy-time quant repair** | 50-100 lines | $5 | -0.001 to -0.005 | medium (rules) |
| 2 | B. NF4-style non-uniform quant | 30-50 lines | $3-5 | -0.001 to -0.003 | low |
| 3 | E. Tensor-train embedding | 100-200 lines | $5 | up to 2 MB freed | high (BPB damage) |
| 4 | C. VQ with shared codebook | 80-150 lines | $5-10 | -0.002 to -0.005 | medium |
| 5 | D. Self-distillation quant | 150-300 lines | multi-day | -0.002 to -0.005 | high |

## Recommended sequencing

1. **A first** — highest EV/risk, mostly already coded, tests rules legality cheaply
2. **B if A doesn't fully land** — orthogonal, also cheap
3. **E if we need huge byte savings** — high-risk swing for big payoff
4. **C/D if multi-day budget allows** — defer to last

## Critical: rules-legality of deploy-time techniques

Before A or any deploy-time technique can ship, must verify:
- Does the challenge allow updating model state at eval time?
- TTT clearly does (it's the established pattern). What's the line?
- AR-self-gen calibration data: no val leak, should be safe
- Using TTT's prefix docs for non-LoRA updates: gray area

Worth a careful read of the challenge rules + maybe asking in the repo
discussions before committing engineering time.
