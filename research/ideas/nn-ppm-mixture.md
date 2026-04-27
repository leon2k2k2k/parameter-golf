# Idea: NN + PPM byte-level mixture — understanding the mechanism

**Date:** 2026-04-27
**Source:** Dissection of #1857 (dexhunter) train_seed42.log diagnostic output. The mechanism is from #1850 (someone114514) → #1857.
**Status:** Understanding-only writeup. No spec yet. Spec 052 will be the actual port.

## The headline finding

The PPM-cluster sub-1.05 numbers don't come from a "better model." They come from **stacking a second compressor downstream of the NN, that does what the NN can't**. Quantitatively, on #1857's run:

```
ppm_full_native tokens=40540160 bytes=151078222
   mix_bpb=1.03176168       ← combined NN+PPM
   ppm_only=2.34028416      ← PPM alone (2.13× WORSE than NN alone!)
   nn_byte_bpb=1.10020163   ← NN alone
   gate_high_frac=0.142408  ← 14.24% of bytes hit the gate
```

The combined predictor is 0.07 BPB *better* than NN alone — using a second predictor that's individually 2.13× *worse*. This is counter-intuitive enough that it's worth understanding deeply.

## Three compressors, three jobs

| Compressor | bpb | What it's good at | What it's bad at |
|---|---:|---|---|
| NN (11L 512d trained on FineWeb) | 1.1002 | Semantic structure: next-sentence prediction, topic flow, syntactic context | Surface byte structure: URL prefixes, digit sequences, code identifiers, repeated formatting |
| PPM-D (order-4, online val-only) | 2.3403 | The 14% of bytes where the previous 4 bytes determine the next byte at >90% confidence | The other 86% of bytes — defaults to ~uniform 8 bpb |
| **NN + PPM mixture** | **1.0318** | **Both** — gate routes each byte to the right specialist | n/a |

**Key insight:** the two compressors are not competing for the same bytes — they're complementary. The gate routes each byte to whichever predictor is more confident.

## The λ-gate — the actual routing logic

For each byte the model is about to score, PPM checks: *given the last 4 bytes, what's the maximum-probability next byte under my order-4 trie?*

- **If that max probability ≥ 0.9** → "PPM is highly confident here." The gate fires. Use λ_lo=0.05 (5% NN weight, 95% PPM weight).
- **Otherwise** → "PPM is uncertain." Use λ_hi=0.9 (90% NN weight, 10% PPM weight).

In #1857's run, the gate fires on **14.24% of bytes**. That's the slice where PPM is doing the heavy lifting.

```
p_mix(byte) = (1 - λ) · p_NN(byte) + λ · p_PPM(byte)
NLL_mix     = -log p_mix(byte)
```

The mixing happens in **probability space**, not log-probability space. This matters because mixing two confident-but-disagreeing predictors gets diluted; mixing one confident with one uniform stays confident.

## Why this works at our scale (16 MB)

The "why is PPM effective" question only makes sense at our parameter budget.

A 100B-parameter LLM has plenty of capacity to memorize that `http://www.` → `g`/`f`/`y`/etc. is much more concentrated than its softmax suggests. It would learn this during training and bake it into the weights — no need for a separate byte-level helper.

At **11L × 512d × 16 MB**, the model can't afford to spend parameters on memorizing surface structure. Every parameter is contested between:
- Semantic prediction (sentence flow, syntax, topic)
- Surface memorization (URLs, numerals, code, formatting)

The training optimizer, given the loss-per-bit incentive, mostly picks semantics — because semantic prediction generalizes better and a 16 MB model can't memorize all surface patterns anyway.

PPM provides surface memorization **for free at eval time**, with online byte-level state that lives in RAM and never enters the artifact. The 14% of bytes where PPM is highly confident are exactly the bytes the NN was forced to under-allocate to.

## Why byte-level (not token-level)

A subtle point that's easy to miss.

The NN scores at the **token alphabet** (8192 tokens). Each token represents 2-8 bytes (a CaseOps-encoded subword). To compute byte-level BPB, you spread the NN's per-token log-probability uniformly across the bytes:

```
p_NN(byte) = p_NN(token)^(1/n_bytes)    # geometric uniform spread
```

This **destroys intra-token byte structure**. Inside a token like `https://www.google.com/abc`, the NN's probability is uniformly spread across all 26 bytes, even though byte-by-byte some are nearly deterministic (`o` after `gle.c`) and some are not.

PPM operates **natively on bytes** — it sees the raw byte stream and captures structure inside tokens that the NN literally cannot represent at byte resolution.

This is also why the mixture is mathematically cleanly defined: both predictors output distributions over the same 256-byte alphabet. The mixing is well-typed.

## The conceptual frame

The NN is a **specialist on long-range semantics**. It's trained on next-token prediction over millions of tokens; it learns syntax, topic, world knowledge, sentence-completion heuristics.

PPM is a **specialist on local byte repetition**. Order-4 byte trie built online. It knows that the byte after `http` is most likely `:` or `s`; it knows that after a `<div ` you'll see attribute names; it knows that inside a number, more digits are likely.

The gate is a **type-classifier**: "Is this byte something a 4-byte byte trie can predict?" If yes, route to PPM. If no, route to NN.

Each compressor handles its specialty. Neither is bad — both are excellent at their thing. The mixture is more accurate than either because the gate routes correctly.

## Implications for our porting effort (informs spec 052)

**1. The win compounds with our other levers.** PPM is downstream of training, downstream of EMA, downstream of quantization. It's also downstream of TTT (which is a *training-time* update of LoRA params, then frozen at scoring time). Adopting PPM doesn't conflict with any of our 047/050 work.

**2. Our NN-only bpb determines the headroom.** If our 050 base post-quant val_bpb is similar to #1857's (~1.10 byte-level), PPM should give us the same ~−0.07 BPB. If we're already lower, the ceiling is closer (less room for PPM to fill). The diagnostic to log on our port: `nn_byte_bpb` and `gate_high_frac` *before* mixing.

**3. The 14% gate-fire rate is an audit signal.** It tells us how much of our val stream has surface structure. If we run our port and see only 5% gate-fire, our val is structurally different from dexhunter's (e.g., our base preprocessing is removing some pattern). Diagnostic worth surfacing.

**4. Tuning levers, in order of likely impact:**
   1. **`threshold` (currently 0.9)** — what counts as "PPM is confident"? Lower = more bytes routed to PPM. Worth a sweep on our val (only takes seconds per setting since the C scorer is fast).
   2. **`λ_lo` (currently 0.05)** — when PPM is confident, how much weight to keep for NN? Currently 5%. Could it go to 1% or 0%?
   3. **PPM `order` (currently 4)** — Howard's sweet spot for English text. Our val is FineWeb after CaseOps tokenization; the byte distribution is shifted. Worth trying order 3, 5.
   4. **`λ_hi` (currently 0.9)** — when PPM is uncertain, how much to keep using PPM? Currently 10%. Going to 0% drops PPM's small contribution on hard bytes.

## Open questions to resolve before speccing 052

- **Does our existing checkpoint (045 armD or 050 base when ready) show the same surface-structure failure pattern as #1857's?** Need to inspect val predictions byte-by-byte. *Pending: `tmp_exec/inspect_val_predictions.py` proposal.*
- **Are our val tokens distributed similarly to dexhunter's?** Both use `fineweb10B_sp8192_caseops` lossless — should be identical, but worth verifying byte counts match.
- **Do we have the OpenMP toolchain on our pod image?** `gcc -fopenmp` needs to be in the eval environment. (Should be — the runpod parameter-golf image has standard build tools.)

## Source data

- `pr-1857:records/track_10min_16mb/2026-04-27_PR1787Base_PPM_OMP_1.0322/train_seed42.log` (the diagnostic line cited above)
- `research/literature/ppm-variants-2026-04-27.md` — background on PPM family + why PPM-D was the right pick
- `pr-analysis/frontier-scan/2026-04-27.md` — cluster analysis context
