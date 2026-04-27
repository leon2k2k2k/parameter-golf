# Idea: Failure analysis — making val_bpb interpretable

**Date:** 2026-04-28
**Status:** Methodology proposal; informs spec 052 (PPM port) and future eval work.
**Companion:** `research/ideas/nn-ppm-mixture.md`

## The problem with val_bpb

A single number — say `val_bpb = 1.0654` — tells us *the model is at this level*. It does not tell us **what to fix**. Two models with the same val_bpb can have radically different failure profiles, and improving one means a different intervention than improving the other.

Concretely: when we say "PPM saves ~0.07 BPB," that win is concentrated on **~14% of bytes**. The other 86% of bytes are unchanged. Without breaking val_bpb apart by *where the loss comes from*, we can't tell:
- Whether PPM will help OUR model (does our 14% match dexhunter's 14%?)
- Whether TTT is recovering a different 14% (probably yes — different failure type)
- Where the next 0.01 BPB win lives (training? quant? eval-time? a different downstream lever?)

## The methodology

For any given model checkpoint, run inference on a chunk of validation data and record per-token diagnostics. Then decompose val_bpb into interpretable buckets.

### What to record per token

For each position `i` in val:
- `actual_token_id` — what the val text says comes next
- `top_k_token_ids` (k=5) — what the model predicted most
- `actual_token_prob` — softmax probability assigned to truth
- `NLL_i` = `-log p_actual` — bits of surprise at this position
- `n_bytes_i` — bytes in the actual token (for byte-level conversion)
- `context_left` — last ~20 chars (for human reading)

### What to compute

**(1) NLL histogram + cumulative contribution**

Bucket positions by NLL and show what fraction of val_bpb each bucket contributes:

```
NLL bucket  |  % positions  |  contribution to val_bpb
0.0–0.5     |    40%        |    10%
0.5–1.5     |    30%        |    25%
1.5–3.0     |    20%        |    32%
3.0–5.0     |     7%        |    18%
5.0+        |     3%        |    15%
```

The bottom two rows (10% of positions, 33% of val_bpb) are where wins live. If PPM addresses the bottom 14%, the question is whether that 14% overlaps mostly with these two rows.

**(2) Top-K worst predictions, with context**

```
NLL=8.3  ctx="...store at "                actual="https"   top1=" the"   p_actual=0.003
NLL=7.5  ctx="...code like def "           actual="parse"   top1=" the"   p_actual=0.007
NLL=6.9  ctx="...released in "             actual="2024"    top1=" the"   p_actual=0.011
```

Reading these is the closest thing we have to introspecting the model. We see exactly *what surprised it* and the surrounding context.

**(3) Per-category stratification**

Tag each position by surface type — URL / NUMERIC / CODE / PROSE / BOUNDARY — and report mean NLL per category:

```
Category       | count   | % of val | mean NLL | contribution to val_bpb
URL-like       |   1,205 |   1.2%   |   4.2    |   5.0%
Numeric        |   2,890 |   2.9%   |   3.8    |  11.0%
Code-like      |   1,876 |   1.9%   |   3.0    |   5.7%
Boundary (BOS) |     510 |   0.5%   |   5.1    |   2.5%
Common prose   |  93,519 |  93.5%   |   0.9    |  84.1%
```

**This is the punchline.** It directly tells us where PPM (URL / NUMERIC / CODE / BOUNDARY) can vs cannot help. If those four categories together are 5-15% of val_bpb, PPM closes most of that gap. If they're only 1%, PPM won't move much.

**(4) Position-trace within a single doc**

Pick one val doc. For every position, show: actual byte, NLL, surrounding context. Highlights *where in a document* failures cluster (start? end? at structural transitions?).

### What the output unlocks

After running this on a model, we can:
- **Estimate PPM gain quantitatively before porting it.** If URL+NUM+CODE+BOUNDARY = ~10% of val_bpb, and PPM closes ~half of those, we'd save ~0.05 BPB. If 20%, we'd save ~0.10.
- **See whether 050's gain over #1736 is from semantic improvement or surface improvement.** Re-run on 050 once it finishes; compare per-category NLL deltas. If the gain is in prose (semantic), it stacks with PPM. If it's in surface, PPM has less headroom.
- **Identify TTT's actual contribution category.** Run with TTT off vs on; see which failure tier shrinks. Hypothesis: TTT recovers some of the boundary tier (per-doc adaptation), not surface.
- **Spot pathological behaviors.** "Why does the model predict ` the` after `released in `?" — that's a model-prior bias that more training won't fix without intervention.

## What this is NOT

- **Not a replacement for val_bpb.** Aggregate val_bpb is still the single comparable number. Failure analysis is interpretation on top.
- **Not a training-time tool.** Run only on saved checkpoints, eval-time only. No backprop.
- **Not a TTT replacement.** Diagnostic, not active.

## Connection to PPM

Per `research/ideas/nn-ppm-mixture.md`, PPM saves bits on bytes where the byte-trie predicts the next byte at >90% confidence (14% of bytes in dexhunter's run). Failure analysis is how we **see what those 14% are** in our val stream. The categorization above (URL / NUMERIC / CODE / BOUNDARY) is roughly that 14%.

If we run failure analysis on our model and find URL/NUMERIC/CODE/BOUNDARY contributes 8% of val_bpb (say), and PPM addresses that with ~50% efficiency, we'd estimate ~0.04 BPB savings from porting PPM. If the output shows 15%, we'd estimate ~0.075 — close to dexhunter's headline number.

## Concrete first run

Target: `runs/047B-loop-kv-shrink-screen/final_model.pt` (most recent finished training; ~1.06-ish; volume mounted on every pod).

Output: `testing/outputs/2026-04-28_047B_failure_analysis/`
- `output.md` — markdown report (histogram, top-20, stratification)
- `per_token.tsv` — raw data dump for follow-on analysis if needed

After running, we read the output together. If the failure profile matches the PPM hypothesis, that's strong evidence to spec 052. If it doesn't (e.g., 047B has very low URL-failure NLL), we re-think.

## What other testing this enables

Once we have `testing/inspect_failures.py` (or whatever we name it), it generalizes:
- Compare two checkpoints by diffing their failure profiles (e.g., 047B vs 050)
- Run on TTT-on vs TTT-off to see where TTT helps
- Run on PPM-on vs PPM-off after spec 052 lands — confirm the win is on the predicted bytes

The methodology is the asset. Each individual run is cheap.
