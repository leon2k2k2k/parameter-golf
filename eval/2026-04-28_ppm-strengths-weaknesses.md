# PPM-D byte-mixture: where it helps, where it hurts, and why

**Date:** 2026-04-28
**Source data:** `testing/outputs/2026-04-28_047B_full_val/ppm_trace.md` (byte-level trace on first 105K bytes of val) + aggregate numbers from `output.md`.
**Setup:** 047B model + PPM-D order=4, λ_hi=0.9, λ_lo=0.05, threshold=0.9. Gate fires when PPM's max-prob over the 256-byte alphabet ≥ 0.9.

This note is a follow-up to `2026-04-28_047B-failure-modes.md` and supersedes the high-level claims in `research/ideas/nn-ppm-mixture.md` with concrete byte-level evidence.

## The mental model

> PPM works as a **doc-level byte memorization plug-in** for any pattern that recurs within val. Character names in fiction, repeated boilerplate, common 4-byte n-grams. The NN can't afford to memorize these at our 16 MB parameter budget; PPM gets them for free at eval time.

## What PPM does well — verified with concrete examples

### Win category 1: Recurring proper nouns

The val set contains an excerpt from Crime and Punishment. Three character names appear dozens of times: `Razumihin`, `Porfiry`, `Raskolnikov`.

| Context | NN top-1 token (prob) | PPM said (prob) | Actual byte | NN bits | Saved |
|---|---|---|---|---:|---:|
| `"why, are you both joking?" razumi` | `_asked` (0.28) | `h` (0.97) | `h` | 9.75 | **−9.63** |
| `"foo! i have muddled it!" por` | `ters` (0.41) | `f` (0.98) | `f` | 8.23 | **−8.13** |
| `s telling me to-day," put in porfir` | `io` (0.35) | `y` (0.96) | `y` | 7.87 | **−7.74** |
| `know very well," he turned to rask` | `ed` (0.74) | `o` (0.98) | `o` | 6.44 | **−6.34** |
| `know very well," he turned to rasko` | `ed` (0.74) | `l` (0.98) | `l` | 6.44 | **−6.34** |

What's happening: NN treats `razumi` like a generic word fragment and predicts common verb continuations (`_asked`, `_io`, etc.) — gives the actual byte `h` only ~0.1% probability. PPM, after seeing the byte sequence `razumi→h` 50 times in val, says `h` with 97% confidence. Gate fires HIGH; mix uses 95% PPM weight. **Net: 9.6 bits saved on a single byte.**

These wins are *huge* per byte (−6 to −10 bits) but rare per stream. Fiction names dominate this category.

### Win category 2: Common phrase byte-level concentration

| Context | NN top-1 token (prob) | PPM said (prob) | Actual | NN bits | Saved |
|---|---|---|---|---:|---:|
| `n in quite a different tone, laughing to` | `_himself` (0.92) | ` ` (0.95) | ` ` | 7.16 | **−7.01** |
| `ard towards men, and valiant against his` | `_own` (0.58) | ` ` (0.95) | ` ` | 7.36 | **−7.22** |
| `r very nature again, and to my mind they` | `_are` (0.19) | ` ` (0.91) | ` ` | 8.70 | **−8.50** |

Subtler mechanism: NN's top-1 token *does* start with the right byte (a space), so at the byte level NN agrees with PPM. But NN's *token-level* probability mass spreads across many multi-byte continuations, so the per-byte NLL on the specific 1-byte target ` ` is high (the spread). PPM at the byte level just predicts ` ` directly with 95% confidence and is right.

### Win category 3: Templated boilerplate

When the val stream contains repeated patterns (forum templates, navigation footers, copyright notices), PPM nails the boilerplate continuation with near-certainty. Each repetition makes PPM more confident. Examples not surfaced in this trace because the slice is small, but the same mechanism as Category 1 applies — count-based memorization of recurring substrings.

## What PPM does badly — verified with concrete examples

### Loss category 1: Multi-byte UTF-8 mid-sequence

PPM operates on raw bytes. Inside multi-byte UTF-8 sequences (curly quotes, Cyrillic, emoji, etc.), the byte distribution is highly context-specific in ways PPM's 4-byte context can mislead.

| Context | PPM thought (prob) | Actual byte | NN bits | PPM bits | Hurt by |
|---|---|---|---:|---:|---:|
| ` i loved everything about it (3:00 �` | `\x81` (0.98) | `\x82` | 0.07 | 30.11 | **+4.32** |
| `find installations of icewarp:"` | ` ` (0.99) | `\xee` | 0.46 | 28.37 | **+4.32** |
| ` ` (mid Cyrillic-quote sequence) | `\xee` (0.91) | `\xc3` | 5.23 | 23.49 | **+4.32** |

PPM was VERY confident on these (0.91-0.99). The actual byte was unrelated. NN got the right answer (0.07-0.46 bits — fine). Gate firing HIGH overruled NN with PPM's 95% weight. The mix paid +4.32 bits per byte.

This is a **structural risk** of high-confidence PPM gating. The 0.9 threshold is conservative precisely to keep these rare.

### Loss category 2: NN was right, PPM hijacked

| Context | NN bits (correct!) | PPM thought (prob) | Actual | Mix bits | Hurt by |
|---|---:|---|---|---:|---:|
| `dditional time in order to interview wit` | 0.54 | `h` (0.97) | `n` | 4.86 | **+4.32** |
| `��where is this article plagiarized from` | 0.01 | ` ` (0.94) | `?` | 4.33 | **+4.32** |
| `bmitted: 2004-09-23 00:00:00 added by` | 0.46 | ` ` (0.93) | `:` | 4.78 | **+4.32** |

Pattern: PPM saw a common 4-byte context (`wit` → `h` for "with"; pre-question-mark space; date colon). PPM's 4-byte view doesn't see that this val instance has a rare continuation. Gate fires; PPM hijacks; loss.

### Aggregate count

From the 105K-byte trace:
- **1924 PPM big-wins** (HIGH regime, saved >1 bit per byte) — mean ~3 bits/byte saved
- **319 PPM losses** (HIGH regime, lost >2 bits per byte) — mean ~4 bits/byte lost
- **Win/loss ratio: ~6:1**

The conservative 0.9 threshold is doing its job. Most HIGH-regime fires are genuine wins.

## Where PPM does NOT help — and why

### Out-of-distribution / OOV continuations (1.6% of val_bpb)

The catastrophic-overconfident-wrong tier from the earlier eval note (e.g., model says `vancou+ver` p=1.0, actual is `vancou+er`).

**PPM doesn't help here.** Both NN and PPM have learned the same dominant pattern from val. When val has a *rare* continuation, both fail together:
- NN top-1 token: `ver` with p=1.0
- PPM context `ncou` → `v` with p=0.95 (also seen "vancouver" many times)
- Gate fires HIGH; both wrong; mix is also wrong

PPM and the catastrophic tail live in different layers of the stack. PPM operates byte-by-byte over local 4-byte contexts; the catastrophic tail is at the *token-prediction* level where the model committed to a wrong word entirely.

### Long-range semantic patterns

PPM only sees the last 4 bytes. Cannot:
- Track topic across a paragraph
- Use entity coreference (subject of sentence is "the doctor" → "she" later)
- Understand syntactic constraints (after a finite verb, no second finite verb)

These are NN's strength. PPM stays out of the way (low confidence, gate doesn't fire) on bytes where semantic reasoning matters.

### Cold start

PPM's tables grow throughout the val stream. At byte 0, gate fires 0%. At byte 105K, ~17%. Full 151M-byte val ends at the converged ~17% rate, but the first ~1M bytes are under-utilized.

This is unavoidable — PPM is online val-only by design (legal under Issue #1017 score-before-update).

## Quantitative win attribution

From the full-val numbers:
- Total PPM gain: **−0.05 BPB × 151M bytes = ~7.5M bits saved**
- Catastrophic-overconfident-wrong tier: 2.26M bits total (PPM doesn't address)
- "Confident-wrong" total available rescue: 0 (PPM also wrong)

Therefore PPM's **−7.5M bits gain comes from the unsure-wrong quadrant** (89% of val_bpb). Most of those bits come from:
1. Recurring proper nouns / boilerplate (Win category 1) — big wins, fewer positions
2. Common phrase byte concentration (Win category 2) — moderate wins, more positions

The categorization-by-regex from `output.md` (URL/NUM/CODE = 2.6% of val_bpb) was a misleading framing. PPM's gain mostly comes from **inside prose**, on bytes where the byte stream has order-4 patterns the NN can't memorize at 36M params.

## Implications for spec 052 (PPM port)

Predicted gain on our 050 base: ~−0.05 BPB, mirroring 047B's gain on the same val.

Sensitivity:
- **Higher gate threshold (0.95)**: fewer wins, fewer losses, potentially small net change. Worth testing.
- **Lower gate threshold (0.85)**: more wins on lower-confidence patterns, more catastrophic-loss exposure. Risky.
- **Higher PPM order (5, 6)**: more memory, slower cold-start, marginally better on rich text. Worth testing on the cached `.npz`.
- **λ_lo = 0.01 (trust PPM more when confident)**: bigger wins per win, bigger losses per loss. Risky.
- **λ_lo = 0.10 (keep more NN)**: smaller wins, smaller losses. Conservative.

Tuning these is a low-cost experiment using the cached NLLs from our run — no GPU needed. Should be a follow-up before finalizing 052.

## The lever's true scope

PPM-D byte mixture is not "URL/NUM/CODE rescue" as we initially framed. It is **online doc-level byte memorization** — a tool for exploiting any byte-level redundancy in val that the NN couldn't afford to learn.

Its gain on this competition (~−0.05 BPB) is **directly tied to how repetitive val is**. Crime and Punishment names + boilerplate make val highly compressible by PPM. A val drawn from a single random article would yield much less.

This is why someone114514's #1850 used a worse base NN (1.087 vs our 0.99) but still landed at 1.005 with PPM: their PPM gain was even larger (−0.082 BPB) because their NN had *more* uncompressed redundancy to harvest. The compounding is bounded by entropy, not by NN quality.

## Open questions

1. **Tune the gate threshold and λ values** on the cached `.npz` — no model rerun needed. Could squeeze 0.005-0.01 more BPB.
2. **Test PPM order 5, 6** — same cached-data approach.
3. **Per-doc-segment PPM gain breakdown** — does PPM's gain concentrate in specific docs (e.g., the C&P excerpt) or distribute evenly? If concentrated, the gain is val-specific and might not transfer to a held-out val.
4. **Combine with TTT** — does PPM stack additively with TTT, or do they overlap on the same unsure-wrong positions? Worth testing.
5. **Simulate "training-data PPM"** — if we built PPM tables from training corpus and shipped them in the artifact, how much extra would we gain? (Likely illegal under Issue #1017 but informative.)

## Source files

- Trace data + concrete examples: `testing/outputs/2026-04-28_047B_full_val/ppm_trace.md`
- Failure analysis (NN-only): `testing/outputs/2026-04-28_047B_full_val/failure_analysis.md`
- Aggregate run report: `testing/outputs/2026-04-28_047B_full_val/output.md`
- Raw cached data: `testing/outputs/2026-04-28_047B_full_val/raw_nlls.npz` (gitignored)
- Tracer script: `testing/trace_ppm.py`
- Implementation: `testing/inspect_with_ppm.py` + `testing/ppm_scorer.c`
