# 047B failure-mode analysis — what does our trained model actually fail at?

**Date:** 2026-04-28
**Source data:** `testing/outputs/2026-04-28_047B_full_val/` (output.md + raw_nlls.npz + failure_analysis.md)
**Setup:** spec 047B (`final_model.pt`, post-EMA pre-quant), full val (40,540,159 tokens / ~151 MB), no TTT, no PPM in this NN-only analysis.
**Methodology:** 4×H100 chunked-2048 with varlen attention via `_build_cu_seqlens`. ~0.012 BPB drift vs official `eval_val` due to eager-vs-compiled forward.

## Headline numbers

```
NN-only val_bpb     ≈ 1.079       (vs official-method 1.06763 on this checkpoint)
top-1 accuracy      = 52.06%
mean NLL/token      = 3.395 bits
```

These are computed across 40.5M val positions. Token-byte conversion uses the val_bytes sidecar.

## The shape of the loss

### 1. The model is well-calibrated

Confidence calibration is unusually clean:

| Top-1 confidence | actual top-1 accuracy |
|---|---:|
| 0.0–0.1 | 6.8% |
| 0.1–0.3 | 19.2% |
| 0.3–0.5 | 38.2% |
| 0.5–0.7 | 58.2% |
| 0.7–0.9 | 79.7% |
| 0.9–0.99 | 95.4% |
| 0.99–1.0 | 99.6% |

When the model says "I'm 90% sure," it's right 95% of the time. When it says "I'm 50% sure," it's right 58%. **The softmax is a meaningful confidence signal**, not a noisy artifact.

### 2. The dominant failure mode is *uncertain-and-wrong*, not *overconfident-and-wrong*

| Quadrant | % positions | mean NLL | % of total val_bpb |
|---|---:|---:|---:|
| Confident-right (`p_top1 > 0.9` & correct) | 23.6% | 0.04 bits | **0.3%** |
| Confident-wrong (`p_top1 > 0.9` & wrong) | 0.7% | 8.10 bits | 1.6% |
| Unsure-right (`p_top1 ≤ 0.9` & correct) | 28.4% | 1.09 bits | 9.1% |
| **Unsure-wrong (`p_top1 ≤ 0.9` & wrong)** | **47.2%** | **6.39 bits** | **89.0%** |

**89% of our val_bpb comes from positions where the model is uncertain and gets it wrong.** The catastrophic overconfident-wrong cases are spectacular but rare (1.6% of total).

This reframes the problem. We're not losing bits to overconfident hallucinations. We're losing bits because the model spreads its probability across many plausible options and the correct answer happens to be a few percent rather than the top of the distribution.

### 3. The long tail dominates

| Worst N% of positions | % of val_bpb |
|---|---:|
| top 1% | 5.0% |
| top 5% | 20.0% |
| **top 10%** | **34.9%** |
| top 20% | 57.8% |
| top 30% | 74.1% |
| top 50% | 92.9% |

The hardest 10% of positions explain a third of total loss. The hardest half explains 93%. The other half is essentially free.

This is the classic shape: **bytes don't contribute uniformly to compression. A small subset is where wins live**.

## The catastrophic overconfident-wrong examples

Even though these are only 0.7% of positions, they're the most informative for understanding *what kind of mistake* the model makes:

| context (last 50 chars) | model said (p) | actually was | bits wasted |
|---|---|---|---:|
| `applying to the vancou` | `'ver'` (1.000) | `'er'` | 35.2 |
| `psychics and astrol` | `'og'` (0.999) | `'gers'` | 37.9 |
| `gardens of rooms and passages in a phalan` | `'x'` (0.998) | `'stery'` | 36.0 |
| `province of sask` | `'at'` (0.999) | `'ache'` | 34.8 |
| `philippine daily inquirer. ang was a juror for the phil` | `'ipp'` (0.990) | `'stage'` | 41.3 |
| `dig out a little hole for each` | `'▁egg'` (0.992) | `'k'` | 35.5 |

**Pattern**: these aren't content surprises. The model knew the *concept* perfectly. What broke was the **tokenization** — the SentencePiece tokenizer split the actual continuation differently than the model expected.

- "vancouver" — model expected `vancou` + `ver`. Actual: `vancou` + `er` (with a different surrounding split).
- "astrologers" — model expected `astrol` + `og` (start of `astrology`). Actual: `astrol` + `gers`.
- "saskatchewan" — model expected `sask` + `at` (start of `saskatchewan`). Actual: `sask` + `ache`.

These are **tokenization surprises**, not knowledge failures. The model knows the word; it's wrong about how the tokenizer splits it.

### Why this matters

- **PPM doesn't help here.** PPM operates on bytes, but these failures are tokens predicting tokens. PPM might smooth the byte-level NLL but the underlying token NLL stays catastrophic.
- **Subword regularization at training time** (BPE-dropout, sampling subword splits during training) would reduce these. We're not doing that.
- **A different tokenizer** (or splitting fewer rare-completion tokens) might also help.

This is a **lever the field hasn't pushed on much** in our trunk-A lineage. Worth flagging as an idea (`research/ideas/` candidate).

## Per-category breakdown

Categories assigned by simple regex on the SP piece:

| Category | % positions | mean NLL/byte | % of val_bpb |
|---|---:|---:|---:|
| PROSE | 86.7% | 1.12 | **94.5%** |
| empty (CaseOps capital markers) | 9.7% | 0 | 2.8% |
| NUMERIC | 3.3% | 2.34 | 2.3% |
| PATH | 0.1% | 3.81 | 0.1% |
| CODE (`;` `{}` etc.) | 0.1% | 4.71 | 0.1% |
| HEX | 0.1% | 1.37 | 0.1% |
| URL | <0.1% | **0.66** | <0.1% |

**PROSE dominates** at 94.5% of val_bpb. The "PPM-addressable" categories (URL/NUMERIC/CODE/HEX/PATH) total only 2.6% of val_bpb.

### The URL surprise

URLs have the **lowest mean NLL** (0.66 bits/byte). This is counter to our PPM hypothesis where we expected URLs to be a NN failure mode.

What's going on: once the model is *inside* a URL (after seeing `http`), the rest is highly templated (`://`, `www`, `.com`, etc.) and the NN handles it fine. The URL *catastrophes* are at **transition points** — where the URL appears in mid-prose:

```
ctx: "diana's untimely death and... she gives to the people"
actual: "▁http"        (URL appearing out of nowhere)
predicted: "▁she" (0.25)
NLL: 24.07 bits
```

The same is true for numbers, dates, code: the byte-level interior is fine; the *boundary* between modes is where the loss concentrates.

## Document boundaries are surprisingly cheap

First byte after BOS: only **1.35 bits/byte** vs overall mean 3.39. **Below average.**

Two possible explanations:
- Many docs start with similar template-like patterns (titles, dates, "Article:", etc.)
- Or the model has overfit to common doc-start prefixes in training

Either way, document boundaries aren't a major loss source. Surprising.

## What this means for downstream levers

### PPM (our current focus)

PPM's gain is reportedly **−0.05 BPB** on this model. Where does it come from?

Mostly NOT from the URL/NUMERIC/CODE categories (only 2.6% of NN val_bpb total — even cutting these in half is 1.3% gain). PPM must be helping inside PROSE — at points where the local 4-byte byte-context concentrates the next-byte probability more than the NN's softmax does.

This explains the gate firing on **17.2% of bytes** (much more than the ~3% surface-structure positions). PPM finds order-4 byte patterns inside prose: common fragments like `tion `, `the `, repeated formatting characters.

### TTT

TTT (per-doc LoRA adaptation at eval time) recovers ~0.013 BPB on this stack. Where does it help?

The **unsure-wrong quadrant (89% of val_bpb)** is the biggest target for TTT. Per-doc adaptation can shift probability mass toward the doc's actual style/vocabulary — turning "spread thin across many options" into "concentrated on the right option."

### Tokenizer-level subword robustness

The catastrophic overconfident-wrong category exposes a **tokenization brittleness**. A model trained with subword regularization would assign lower probability to `vancou+ver` (one specific split) and more to alternatives, smoothing the catastrophic NLL. This is a training-time intervention; out of scope for our 3-day window but worth noting.

### Specs to consider

Based on this analysis, the highest-EV intervention concepts (in decreasing order):

1. **PPM-D port** — already planned (spec 052). Should give ~−0.05 BPB. Most of this gain comes from inside-prose 4-byte patterns, not URL/NUM/CODE specifically.
2. **TTT tuning** — TTT_LORA_RANK=80 (#1855) already on our list. Targets the unsure-wrong quadrant.
3. **Subword regularization** (NEW idea, not yet specced) — train with BPE-dropout or sampling-based subword splits. Should attack the catastrophic overconfident-wrong examples directly. Estimated: ~−0.001 to −0.003 BPB. Worth a spec.

## What's NOT a problem

Things we expected to be issues that turn out not to be:
- **Document boundaries** — easier than average, not harder. Skip.
- **Numbers/code structure within a sequence** — handled fine. The catastrophes are at *transitions*, not interiors.
- **Model overconfidence** — calibration is clean. We're not losing many bits to confident hallucinations.
- **URLs** — lowest mean NLL of all categories. Not a problem class.

## Open questions for follow-up

1. **Sub-categorize PROSE more carefully**: separate "first byte of mode-switch" from "mid-content" from "word-completion." Likely most of the unsure-wrong loss concentrates in mode-switch positions.
2. **Quantify catastrophic overconfident-wrong by tokenization split**: does it correlate with rare bigrams in the SP vocab? (Would suggest training-time tokenizer mitigation.)
3. **Compare 047B → 050 once 050 finishes**: is 050's improvement from PROSE, surface, or something else? Would clarify whether the BOS-fix specifically helps.
4. **Run failure analysis with TTT on**: which quadrant does TTT actually rescue? (Need a TTT-on checkpoint to test.)

## Source files

- Raw data: `testing/outputs/2026-04-28_047B_full_val/raw_nlls.npz` (1.1 GB, gitignored)
- NN-only + PPM mix headline: `testing/outputs/2026-04-28_047B_full_val/output.md`
- This eval note's raw analysis: `testing/outputs/2026-04-28_047B_full_val/failure_analysis.md`
- Implementation script: `testing/inspect_with_ppm.py`
- Post-hoc analysis script: `testing/analyze_failures.py` (operates on cached `.npz`)
