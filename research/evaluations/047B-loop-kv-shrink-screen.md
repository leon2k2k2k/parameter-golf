# Evaluation — Spec 047B (loop-layer KV head shrink, LOOP_LAYER_NUM_KV_HEADS=2)

**Run dir:** `runs/047B-loop-kv-shrink-screen/`  
**Commit:** `02977b7` on `exp/047B-loop-kv-shrink`  
**Baseline:** spec 045 armD (AC-fix), pre-quant EMA **1.06479**, post-quant **1.07387**  
**Eval date:** 2026-04-27

## Result — iterate band, surprising quality resilience

| metric | AC-fix baseline | 047B | Δ |
|---|---|---|---|
| pre-quant EMA val_bpb | 1.06479 | **1.06763** | **+0.00284** |
| post-quant val_bpb | 1.07387 | **1.07653** | **+0.00266** |
| quant cost (Δ pre→post) | +0.00908 | +0.00890 | −0.00018 (healthier) |
| submission size | ~15.96 MB | 16.30 MB | **over cap** |
| steps in budget | ~4500 | 5213 | +15% (faster loop) |
| tok/s post-loop-activation | ~3.82M | ~4.19M | +10% |
| accept gate | — | ≤ 1.06679 | **missed by +0.0008** |
| kill gate | — | > 1.06979 | **clear — not killed** |

**Verdict: KILL (size) + informative iterate (quality).** Pre-quant bpb lands in the iterate band — above accept (+0.00284 > +0.002), below kill gate (1.06979). But the submission size (16.30 MB) is already over the 16,000,000-byte hard cap, making this a non-starter for submission regardless of bpb.

## "Attention reduction isn't too bad" — the main finding

Halving KV heads on only the 3 loop layers (layers 3–5, running 3× per forward = 9 effective attention ops) costs **+0.0028 bpb**. For reference:
- The loop band is where the model spends the most attention compute (3× reuse).
- ALBERT (1909.11942) found FFN tying is the dominant quality cost in tied transformers; attention tying is approximately free.
- This result is consistent: even aggressive KV compression on the recurrent band is survivable.
- Per effective-attention-op damage: +0.00284 / 9 ≈ 0.0003 bpb per attention op neutered — extremely mild.

The quality signal is real and positive. The execution failure is purely mechanical (size).

## Why the size didn't improve

The implementation kept the full 22-row `kv_bank` intact and added `loop_kv_bank` on top. The loop rows of `kv_bank` are never written at runtime (they receive no gradient through `_bank_weights`) but they still exist in the serialized state dict and get quantized and compressed. Net effect: the compressed artifact grew, not shrank.

To realize the size benefit, the `kv_bank` shape must be reduced to exclude loop rows — i.e., reconstruct it as an `[N - num_looped, kv_dim, model_dim]` parameter. The GPTQ round-trip wiring already handles the split correctly; the missing piece is shrinking the source bank.

## Throughput gain

047B is ~10% faster per step post-loop-activation (4.19M vs 3.82M tok/s at step 2500). This was expected — smaller KV projections reduce the flash-attn memory bandwidth for the 3 loop layers. The +15% step count (5213 vs ~4500) in the same wallclock budget means 047B is strictly cheaper to train per unit quality than this result suggests. At matched-step comparison the bpb delta would be slightly larger, but the result is still in the iterate band.

## GPTQ health

Post-quant degradation for 047B: +0.00890 (1.06763 → 1.07653). Baseline: +0.00908. Essentially identical — confirms that `loop_kv_bank` quantizes as cleanly as the regular KV projections through the GPTQ pipeline wired in `02977b7`.

## Decision — KILL this variant; iterate on the mechanism

1. **This variant: KILL.** Size exceeds cap; bpb misses accept gate.
2. **Mechanism: PROMISING — pursue 047 (global KV=2) as the size-saving path.**  
   Global `NUM_KV_HEADS=2` affects all 11 layers (not just 3), giving proportionally larger size savings and a clean implementation (no kv_bank surgery needed). The 047B quality result (+0.0028 for 3 layers) implies a global KV=2 might cost +0.008–0.01 bpb on all layers but save ~30% of KV param bytes. Whether that's a win depends on how much headroom the size saving buys for other levers.
3. **If pursuing loop-layer-only KV shrink:** the fix is to reduce the kv_bank shape to `[N - num_looped, ...]` and reconstruct the serialized form excluding loop rows. This requires a non-trivial `_rebank_state_dict` change but the GPTQ dispatch logic is already correct.
4. **Throughput bonus worth banking.** The 10% per-step speedup at the loop stage is real. If the loop FFN expressivity work (047C/D) can recover the bpb, a combined KV-shrink + AdaLN variant would be faster AND better.

## Next steps

- Launch **047** (global NUM_KV_HEADS=2) — already specced and launch script ready at `tmp_exec/launch_047_kv2.sh`. Assess whether the size gain justifies the quality cost at 11 layers.
- Run **047D** (AdaLN per-pass conditioning) — TRAINING_ONLY_SCREEN, 20min. If it clears −0.001 bpb it plausibly compensates the KV-shrink penalty.
- Note to future specs: if attempting loop-layer-only KV reduction again, shrink kv_bank shape rather than adding a second bank.

## Cost

~$1.50 (4×H100, ~45 min including prewarm). Pod stash at `/workspace/.inductor_cache_02977b7_loop_kv2` eliminates prewarm for future 047B-variant runs.

## Cross-references

- Spec: `research/specs/047B-loop-kv-shrink-screen.md`
- Idea: `research/ideas/loop-ffn-expressivity.md`
- Next: 047 (global KV=2), 047D (AdaLN), 047C (per-pass LoRA)
