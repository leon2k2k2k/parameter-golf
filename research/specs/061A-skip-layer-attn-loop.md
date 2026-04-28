# Spec 061A — Skip-Layer Attention adapted to loop block

**Status: DRAFT — implementation deferred.** Code change requires threading
cached K,V through `Block.forward` across the encoder loop's expanded
index list, which has non-trivial torch.compile interactions (new graph
variants on cache hit/miss; cross-block tensor liveness). Proper
always-tensor scaffolding + a 2×H100 smoke must precede pinning a
commit. Recommend a follow-up implementation session before handing
off. **Do not launch a pod against this spec — there is no pinned
commit yet.**

**Date:** 2026-04-29
**Branch:** `exp/061A-skip-layer-attn-loop` (not yet created)
**Pinned commit:** **TBD** — pending implementation pass
**Parent:** 060A (#1855 port; would fork from `exp/060-resume-ckpt @ a0a48b7`).

## Note on existing architecture (added 2026-04-29 after code review)

The 060A baseline **already implements a U-Net structure** at the
*hidden-state* level: `encoder_indices` push to a `skips` stack,
`decoder_indices` pop and combine via `skip_weights` (per-channel) and
optional `skip_gates` (per-channel sigmoid lerp; `SKIP_GATES_ENABLED=1`
by default). This is residual-stream skip, which is **not** the same
as Skip-Layer Attention.

This spec adds an *attention-level* skip orthogonal to the existing
hidden-state skip: in the last visit of loop layer 5 in the decoder, a
fraction of attention heads use Q from the current residual but K,V
*projected from* the input residual at the matched encoder visit of
loop layer 3 (or alternative source — see open questions). The
existing `skip_weights` pass full hidden state forward; this new
mechanism gives attention selective access to earlier representations
without polluting the residual stream itself.

## Hypothesis

In Skip-Layer Attention (Peng et al., arXiv 2406.11274), GPT-2 124M @ 16k context
gained **+0.1329 absolute loss** over baseline by routing ~3/4 of attention heads
in layer ℓ to use K,V from layer ℓ−n (zero new parameters; pure routing). We
adapt this to our looped block: in the **last loop pass only**, designated heads
of loop-layer-5's attention compute Q on the current residual but use K,V
projected from the residual at the **input of loop-layer-3** (within the same
pass). This is a within-pass cross-layer skip — no cross-pass cache needed.

Mechanism rationale: by pass N−1, the residual has accumulated through 3 loop
layers × N passes of refinement. Skip-routing recovers earlier-stage
representations at zero parameter cost, which is exactly what the GPT-2 finding
attributed the gain to.

This is also the cheapest novel-to-this-stack untie of the recurrence we found
in literature review. Our 98 KB byte budget is *not* spent here — preserved for
a future spec if 061 lands.

## Baseline

060A (#1855 port, single seed 42).

## Expected Δ

**−0.001 to −0.005 BPB** vs 060A.

Confidence: medium. The +0.13 loss finding at GPT-2 124M is well-validated, but
transfer to our depth-recurrent ~30M model is unknown. Their result *grew with
sequence length* — our short-sequence training regime may favor or hurt us
depending on the actual seq-len signal in our val set.

## Accept criteria

- post-quant + post-TTT val_bpb ≤ 060A − 0.0005 (1-seed screen)
- artifact ≤ 16 MB (no new params; byte budget unchanged)
- no NaN, no compile pathology, no mid-run recompile

## Config diff vs 060A

```
SKIP_LAYER_ATTN_ENABLED = 1
SKIP_LAYER_ATTN_NUM_HEADS = 4   # of 8 total heads — 50%
SKIP_LAYER_ATTN_PASSES     = "last"   # only fire on last pass
SKIP_LAYER_ATTN_FROM_LAYER = 3   # source loop layer (K,V cached here)
SKIP_LAYER_ATTN_AT_LAYER   = 5   # target loop layer (heads use cached K,V)
```

## Code changes

- At loop-layer-3 attention forward, cache `K_3, V_3` (shape `[B, H_skip, T, d_h]`)
  on the loop block module. Identity-cost on every pass; reused on pass N−1.
- At loop-layer-5 attention forward, when `pass == N−1` and
  `SKIP_LAYER_ATTN_ENABLED`, splice the first `SKIP_LAYER_ATTN_NUM_HEADS` heads'
  K,V to use the cached `K_3, V_3` instead of recomputing from current residual.
- Always-tensor pattern (per `feedback_always_tensor_block_kwargs`): a binary
  multiplier tensor selects cached vs fresh K,V — no Python `if` inside compile
  region. Single graph variant.
- Compile checklist (per `feedback_spec_compile_checklist`): no weight slicing,
  no None passthrough, identity buffers initialized.
- Init: `SKIP_LAYER_ATTN_ENABLED=0` ⇒ identical bytes/behavior to 060A.

Branch `exp/061A-skip-layer-attn-loop` (not yet created) from
`exp/060-resume-ckpt @ a0a48b7`. Commit hash: **TBD** — research must
implement, push, and pin before mini rung.

**Implementation surface:** add a `kv_skip_cache` keyword to
`Block.forward` (always-tensor: zeros buffer when inactive); on the
encoder-side visit of `LOOP_START` layer, populate the cache; on the
decoder-side visit of `LOOP_END` layer's last pass, splice the cached
K,V into `SKIP_LAYER_ATTN_NUM_HEADS` of the heads. Both `_forward_hidden`
and `forward_ttt` paths must mirror the change. Compile graphs to
verify: (1) baseline (no cache), (2) cache populate, (3) cache
consume; ideally the always-tensor pattern collapses (2)+(3) to one
variant that's always-on with zero-cache acting as identity.

## Hardware ladder

- **Mini rung: 2×H100, 5-min full-arch smoke.** Required (code change). Verify:
  no compile pathology, throughput within 3% of 060A baseline at matched step.
- **Official rung: 8×H100, 1 seed (42)**, 600s train + 600s eval.

If mini throughput cost > 5%, halt and reconsider head fraction.

## Seed plan

1 seed (42) for screening. Multi-seed only if win clears noise floor (~0.0007).

## Inputs

- Train/val/tokenizer: standard 060A paths
- Hotstart: none (fresh training)
- Region: NE-1 first per memory; JP fallback

## Checkpoints emitted

- `final_model.pt` — pre-quant post-EMA
- `final_model.int6.ptz` — post-quant submission
- `train.log` — full log including phased TTT eval line

Saved to `/workspace/runs/061-skip-layer-attn-loop/seed_42/`.

## Stop-early criteria

- train_loss > 5.0 at step 1000 → kill
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- mid-run torch.compile recompile → kill (always-tensor violated)
- mini rung shows >5% throughput cost → halt before official rung
- Compile fail → kill, debug

## Cost estimate

~$1 mini + ~$5 official = **~$6**.

## Open questions for interview

1. **Source choice.** Spec uses *within-pass cross-layer* (layer 3 → layer 5).
   The other readable adaptation is *across-pass same-layer* (pass-0 layer-5 →
   pass-N−1 layer-5). Within-pass is cleaner (no cross-pass state). If 061
   lands flat, 062b would test the across-pass variant.
2. **Head fraction.** Paper's best was ~3/4 of heads. We default to 1/2 to
   minimize blast radius. If win, sweep 3/4 in followup.
3. **What halts** if mini reveals an attention-shape mismatch (our K_3 may be
   shaped for fewer heads if we use GQA/MQA)? Halt and ask.
4. **Compose with 062?** If both individually clean, freeze 063-combined.
   If only 061 lands, that's a clear independent spec.

## Followups (NOT in this spec)

- 061b: if 061 wins, sweep `SKIP_LAYER_ATTN_NUM_HEADS ∈ {3, 5, 6}`.
- 062: per-pass output gate (independent intervention, separate spec).
- 063-combined: 061 + 062 if both individually clean.
