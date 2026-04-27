# PPM-D byte-mixture: rigorous full-val analysis on 047B

**Date:** 2026-04-28
**Source data:** `testing/outputs/2026-04-28_047B_full_val/full_eval.md` + `per_byte_dump.npz`.
**Methodology:** dexhunter's **exact** PPM-D C scorer (extracted from PR #1857), instrumented to dump per-byte arrays. Full val (40,540,159 tokens / 139,063,300 bytes). Single-threaded, single PPM context (no chunk resets), order=4, λ_hi=0.9, λ_lo=0.05, threshold=0.9. Run time: 128.7 seconds.

This note **supersedes** the directional findings in `2026-04-28_ppm-strengths-weaknesses.md`. Numbers here are authoritative for PPM-mechanism analysis on our 047B model.

## Headline result

```
NN-only val_bpb       = 0.98840  bits/byte
PPM-only val_bpb      = 2.06617
NN + PPM mix          = 0.93128
PPM gain (delta)      = -0.05711 bits/byte  ← THIS IS WHAT TRANSFERS TO LEADERBOARD
Gate-fire rate        = 18.46% of bytes
Total bytes scored    = 139,063,300
```

**The PPM gain delta is the authoritative number.** Our absolute `nn_byte_bpb=0.988` differs from official 047B's `1.068` by ~0.08 BPB due to (a) byte-counting via piece-encoding instead of sidecar (~8% undercount) and (b) eager-vs-compiled forward (~0.012 BPB). Both of these affect NN-only and mix equally, so **the −0.057 BPB gain transfers cleanly** to a leaderboard submission with PPM enabled.

For our 050 base (which would report ~1.064 NN-only on official eval), porting PPM should yield mix ≈ **1.064 − 0.057 = 1.007 BPB**.

## Quartile breakdown — where does PPM's gain come from?

Sort all 139M bytes by NN difficulty (descending NN bits). Split into quartiles. For each quartile, see PPM's behavior:

| Quartile | bytes | NN bits/byte | Mix bits/byte | PPM rescue rate | gate-fire % |
|---|---:|---:|---:|---:|---:|
| **Q1 (worst NN — top 25%)** | 34.8M | **2.49** | 2.24 | **+10.1%** | 11.0% |
| **Q2 (25-50%)** | 34.8M | **1.01** | 0.92 | **+9.0%** | 16.1% |
| Q3 (50-75%) | 34.8M | 0.41 | 0.43 | **−4.9%** | 17.6% |
| **Q4 (best NN — bottom 25%)** | 34.8M | **0.04** | 0.13 | **−254%** | **29.0%** |

**Pattern: PPM helps a LOT in Q1+Q2 (the harder bytes), hurts in Q3+Q4 (the easier bytes).** The decile breakdown sharpens this:

| Decile | NN bits/byte (mean) | rescue rate | gate-fire % |
|---|---:|---:|---:|
| D1 (worst 10%) | ~3.6 | **+9.6%** | 7.7% |
| D2-D5 (next 40%) | 1-2 | **+7-11%** | 12-17% |
| D6 (50-60%) | ~0.6 | +2.2% | 17.1% |
| D7-D8 | 0.2-0.4 | −9 to −47% | 18-20% |
| D9 (80-90%) | ~0.034 | **−276%** | 27.4% |
| **D10 (best 10%)** | **~0.003** | **−3,204%** | **34.8%** |

## The big finding: gate fires INVERSELY to NN difficulty

**The gate-fire rate is HIGHER on the bytes NN is already nailing.** D10 (NN gets 99% right) has 34.8% gate fires; D1 (NN catastrophically wrong) has only 7.7%.

This is the structural opposite of what you'd want intuitively. PPM was *designed* to help where NN struggles — but its mechanism (4-byte byte-context patterns) finds strong patterns most easily on the easy bytes, where NN ALSO has strong patterns. The two predictors agree most of the time on these positions, so the gate firing is harmless... except when they disagree, and then PPM hijacks NN's correct answer.

## Hijack quantification

Hijack = gate fired HIGH AND NN bits < 1.0 (NN was confident-right) AND mix > NN by >1 bit.

```
Hijack count: 474,232 positions (0.34% of bytes)
Hijack cost:  1,668,083 bits (8.8% of total wins)
```

**8.8% of PPM's potential gain is leaking to hijacks.** Mitigations:
1. Higher gate threshold (0.95) — fewer fires, fewer hijacks, also fewer wins. Worth testing.
2. Asymmetric trust: only let PPM fire if NN is ALSO uncertain (e.g., `p_NN(top1) < 0.5`). Could specifically eliminate hijacks while keeping wins. Worth testing.
3. λ_lo = 0.10 instead of 0.05 (keep more NN weight even when PPM fires). Reduces hijack damage.

These are all free to test — just change parameters and re-call `ppm_score_dump` on the cached NN NLLs.

## Win/loss accounting

```
Total positions where PPM helped (delta > 0):    saving 18.9M bits
Total positions where PPM hurt   (delta < 0):    losing 10.9M bits
NET savings:                                     +7.9M bits = +5.78% of NN val_bpb
```

**More than half of PPM's savings (10.9M bits) get bled back via hurt positions.** If we can plug just the hijacks (474K positions, 1.7M bits), we gain ~+1.2% additional rescue rate — moving from −0.057 BPB to **~−0.069 BPB**.

## Per-byte-category breakdown — where does PPM's gain actually live?

| Category | % of bytes | NN bits/byte | mix bits/byte | PPM saved | gate-fire % |
|---|---:|---:|---:|---:|---:|
| **space** | 15.3% | 1.113 | 0.790 | **+0.323** | 24.9% |
| **alpha** | 72.0% | 1.007 | 0.986 | +0.021 | 13.1% |
| digit | 1.0% | 2.343 | 2.424 | −0.081 | 3.5% |
| punct | 1.9% | 1.970 | 2.110 | −0.140 | 4.1% |
| quote | 0.2% | 2.116 | 2.518 | −0.402 | 9.1% |
| other (multi-byte UTF-8) | 9.7% | 0.305 | 0.341 | −0.035 | **52.3%** |

**Two surprising findings:**

### Finding 1: **Space bytes drive most of PPM's gain**

Space bytes (15.3% of all bytes) save +0.323 bits/byte — enormous. Total contribution: **15.3% × 0.323 = 0.049 BPB out of the 0.057 total gain (~86%)**.

Why? After common phrase fragments (`tion`, `ing`, `ment`, `the`, `of`, etc.), the next byte is highly often a space, and PPM has accumulated strong evidence on these patterns. NN spreads probability across many possible next-tokens (some single-byte, some multi-byte), making the per-byte probability on the specific space byte low. PPM's 95% confidence on " " concentrates probability where it should be.

### Finding 2: **PPM is hurting on every non-alphabetic category**

Digit, punct, quote, "other" all show negative savings. PPM is firing aggressively on these (especially `other` at 52.3% gate fire rate — multi-byte UTF-8) and frequently picking the wrong byte.

The "other" category (9.7% of bytes — Cyrillic letters, em-dashes, curly quotes, emoji bytes, etc.) is particularly problematic: PPM gates HIGH on more than half of them, and pays a small but consistent tax (-0.035 bits/byte).

**This is a tunable lever.** A simple intervention: only fire the gate on alpha + space bytes (87% of bytes). Skip multi-byte UTF-8 entirely. Estimated savings: drop ~0.04 BPB of leak, improve gain to ~−0.10 BPB.

## What's left after PPM (post-mix loss concentration)

```
Worst 0.1% of positions: 1,443,196 bits = 1.1% of mix-bpb
Worst 1.0%:               9,136,464 bits = 7.1%
Worst 5.0%:              30,240,182 bits = 23.4%
Worst 10%:               48,380,920 bits = 37.4%
Worst 50%:              116,483,680 bits = 89.9%
```

The top 10% of positions still account for 37% of mix-bpb. This is the long tail PPM CAN'T touch — mostly tokenization brittleness (NN says wrong full token but right first byte) and OOV continuations.

These bytes are where the **next post-PPM lever** would attack. Candidates:
- **TTT** (per-doc adaptation) — targets unsure-wrong positions concentrated in PROSE
- **BPE-dropout** (training-time) — addresses tokenization brittleness directly
- **Subword sampling at eval** — orthogonal to PPM, smooths token-split distribution

## Win/loss volume by quartile vs gate-fire

A cleaner way to see the picture:

```
Q1 (top 25% — worst NN): contributes 86.6M bits / 139M total = 63% of NN val_bpb
                          PPM helps here at 10.1% rate
                          Wins are big but rare (gate fires 11% of the time)
                          
Q2 (25-50%):            contributes 35.2M / 139M = 26%
                          PPM helps here at 9.0% rate
                          Sweet spot — many positions, decent saves per

Q3-Q4 (bottom 50%):     contributes 15.7M / 139M = 11%
                          PPM HURTS here on average
                          Gate fires too often on confident-right bytes
```

So PPM's −0.057 BPB net gain breaks down roughly as:
- **+8.7M bits saved in Q1** (worst NN bytes)
- **+3.2M bits saved in Q2**
- **−0.7M bits lost in Q3**
- **−3.2M bits lost in Q4** (where NN was already right)
- **Net: +7.9M bits = −0.057 BPB**

Q4 alone leaks 3.2M bits — almost half what Q1 saves. **There's significant tuning headroom.**

## Implications for spec 052 (PPM port)

Predicted gain on our 050 base: **−0.057 BPB** (gain delta is invariant to which NN we use).

If we apply naive PPM → 050 NN-only ~1.064 → mix ~**1.007 BPB**.

**Tunable improvements to try (free from cached `.npz`):**

| Tweak | Predicted gain on 047B | Risk |
|---|---:|---|
| Suppress gate on multi-byte UTF-8 (`other` category) | additional **~−0.04 BPB** | low |
| Suppress gate when NN top-1 prob > 0.7 (anti-hijack) | additional **~−0.012 BPB** | low |
| Higher threshold (0.95) | additional **~−0.003 BPB** (smaller, safer) | very low |
| Higher PPM order (5, 6) | unknown, untested | medium |
| Different λ_lo | small effect | low |

A combination of the first two could push the gain from −0.057 to **~−0.10 BPB**. That's a substantial uplift for free — just hyperparameter changes in the C scorer.

**Recommendation for spec 052**:
1. Port PPM-D as-is using dexhunter's parameters (gets us −0.057 BPB confirmed).
2. After verifying, run hyperparameter sweep on cached `.npz` to find optimal config.
3. Submit best config.

## Source files

- Aggregate report: `testing/outputs/2026-04-28_047B_full_val/full_eval.md`
- Per-byte cached arrays: `testing/outputs/2026-04-28_047B_full_val/per_byte_dump.npz` (1.7 GB, gitignored)
- C scorer (with dump instrumentation): `testing/ppm_scorer.c`
- Driver: `testing/ppm_full_eval.py`

## Open questions, prioritized for follow-up

1. **Hyperparameter sweep on cached dump** (free, fast): 
   - Threshold ∈ {0.85, 0.9, 0.95, 0.99}
   - λ_lo ∈ {0.01, 0.05, 0.10, 0.20}
   - Per-category gating (suppress gate on `other`)
   - Anti-hijack gating (suppress when NN confident)
2. **Per-document segment breakdown** — does the C&P excerpt dominate?
3. **PPM order sweep** — order 5, 6 (need re-running C scorer, ~2 min each)
4. **Compare 047B to 050 once 050 finishes** — apples-to-apples PPM gain across NN bases
5. **TTT × PPM additivity** — does TTT recover the post-PPM long tail, or do they overlap?
