# Spec 046I — compression-side byte savings (concat tensors + brotli tuning + zstd)

**Slug:** `046I-compression-improvements`
**Created:** 2026-04-27
**Status:** READY (Q-J), DRAFT for Q-I and Q-K (need ~80-150 lines code)
**Branch:** `exp/046-quant-repair`
**Commit:** TBD (after code lands for Q-I, Q-K)
**Parent:** `research/ideas/quant-repair.md` (Tier 1 byte-savers Q-I/Q-J/Q-K)

## Hypothesis

We're packed at 15.95 MB / 16.00 MB cap with only 46 KB headroom. SDClip wins
(046B-tight -0.00086, 046G-tightest -0.00216) all overflow by 184-689 KB. We
need ~50-200 KB of compression-side savings to ship even the smallest win.

Three independent compression-side levers:

**Q-I — Concatenated tensor compression**: brotli currently compresses each
tensor separately. Concatenating similar tensors (e.g., all 11 MLP fc weights)
before compression lets brotli exploit cross-tensor statistics (shared scale
distributions, repeated patterns). Lossless transformation.

**Q-J — Brotli quality + custom dictionary**: verify brotli is at max quality
(Q=11). Optional: train a content-specific dictionary from typical quant'd
LLM weight statistics.

**Q-K — ZSTD with trained dictionary**: switch compressor to ZSTD with a
content-trained dictionary. ZSTD often beats brotli for structured data when
paired with a domain dictionary.

## Arms

### Phase 1 — sanity & cheapest wins (Q-J)

| Arm | Change | Code |
|---|---|---|
| **046I-brotli-verify** | Confirm brotli quality is at max (Q=11). If not, bump it. | ~5 lines |
| **046I-brotli-trained-dict** | Add a 4-16 KB pre-shared dictionary trained on typical quant outputs | ~20 lines + 1 dict file |

Phase 1 cost: ~$2.

### Phase 2 — concat tensors (Q-I)

| Arm | Change | Code |
|---|---|---|
| **046I-concat-mlp-fc** | Concatenate all 11 MLP fc int6 weights before brotli; store offsets in metadata | ~40 lines |
| **046I-concat-attn** | Concatenate all attn matrices (c_q/k/v/proj × 11 layers) before brotli | ~40 lines |
| **046I-concat-all-int6** | Concatenate ALL int6-quantized tensors into one compression unit | ~50 lines |

Phase 2 cost: ~$3.

### Phase 3 — ZSTD (Q-K)

| Arm | Change | Code |
|---|---|---|
| **046I-zstd-default** | Switch compressor to ZSTD level 22, no dict | ~10 lines |
| **046I-zstd-dict** | ZSTD level 22 with trained dictionary | ~50 lines + dict |
| **046I-zstd-concat** | ZSTD + concatenated tensors (combine Q-I + Q-K) | ~70 lines + dict |

Phase 3 cost: ~$3.

## Predicted byte savings

| Variant | Optimistic | Realistic | Pessimistic |
|---|---|---|---|
| brotli-verify | +0 | +0 | +0 |
| brotli-dict | +30 KB | +15 KB | +5 KB |
| concat-mlp-fc | +50 KB | +20 KB | +5 KB |
| concat-all | +100 KB | +40 KB | +10 KB |
| zstd-dict | +50 KB | +20 KB | -10 KB (worse than brotli) |
| zstd-concat | +150 KB | +60 KB | +10 KB |

**Stacking best variants: target 50-200 KB freed.**

## Acceptance

For each arm:
- **Win**: net byte savings > 30 KB at unchanged val_bpb
- **Marginal**: 5-30 KB savings
- **No effect**: ±5 KB
- **Loss**: < -5 KB

Then test stacking the winners with 046G-embed-only-tighter (+7 KB needed) and
046B-tight (+184 KB needed) to see what becomes legal-shippable.

## Risk

All compression-side changes are **lossless** — same val_bpb expected.
Risks are:
- Code bugs (test deserialize round-trip)
- Subtle metadata format issues
- Compression that "works on this seed but not others" — unlikely for lossless ops

## Cost

Total ~$8 across all 3 phases, ~1-1.5 hours wallclock.

## Dependencies

- Q-J Phase 1 has no code dependency, can run immediately
- Q-I and Q-K need code changes (~30 min - 1 hour each)
- Trained dictionaries (for Q-J/Q-K dict variants): need to extract a sample of typical quant outputs from prior runs; ~15 min

## What to watch

- **`Total submission size quantized+brotli:`** in log — the headline number
- **deserialize correctness**: re-running quantized eval should give identical val_bpb (lossless = identical)
- **compression time**: zstd level 22 is slow; verify it fits in serialize budget (~1-2s)
