# 2026-04-29 frontier scan (incremental)

**Baseline:** PR #1736 @ 1.06549. Working research baseline post-#1902 = #1855 (codemath3000) @ 1.06108.
**Last scan:** 2026-04-29T22:00Z (state file forward-dated; effective scan boundary is ~PR #1911).
**PRs processed:** 21 new (#1906-#1931 minus already-scanned). 0 status flips on existing entries (open prior set unchanged).

## New record-track PRs

| # | author | claimed_bpb | verdict | legitimacy | builds_on | trunk | note |
|---|---|---|---|---|---|---|---|
| 1906 | AayushBaniya2006 | 1.06136 | clean | disputed | 1797 | B | repro of suspended #1797 — inherits validity-pending |
| 1907 | GodlyDonuts | 1.10705 | broken | broken | 1874 | D | Newton-Muon negative result (forward-hook recompile saturation) |
| 1908 | romeerp | **1.06081** | clean | likely-legal | 1855 | B | AWQ-style activation-aware mixed-bit GPTQ (top-1 group int8, rest int6) |
| 1909 | GodlyDonuts | 1.06996 | clean | likely-legal | 1874 | D | LoRA TTT rank 128→192 — within noise |
| 1910 | hardik-bhalekar | 2.0237 | broken | broken | unknown | unknown | ternary blowup |
| 1912 | WesSwogger | 1.31630 | other | unknown | unknown | unknown | export plumbing draft |
| 1913 | Jeffrey-Le | 1.0847 | clean | likely-legal | (older SP8192) | A | multi-temperature AR GPTQ calibration |
| 1915 | AidenGeunGeun | 1.06504 | clean | likely-legal | (CaseOps+LQER, no MIN_LR) | A | per-doc LoRA TTT with no global SGD; per-doc reset + length bucketing |
| 1916 | Christopher-Lee-McClendon | n/a | other | legal | 1851/1868/1873/1876 | new | non-record PPM-D legality framework (responding to Issue #1872) |
| 1917 | Blitzo125 | 1.1636 | other | unknown | unknown | unknown | weak BigramHash |
| 1918 | aquariouseworkman | **1.06086** | clean | likely-legal | 1855, 1908 | B | AWQ-lite (same pattern as #1908; 0.00022 below #1855, within seed std) |
| 1919 | dev-pratap-singh | 1.0587 (UNVERIFIED) | other | unknown | 1855/1797/1767 | B | ParResid + DR + LoRA TTT + int4/6/8 mixed + AWQ stack — author admits no H100 verification |
| 1921 | oresamadom | 4.916 | other | legal | unknown | unknown | RTX2070 smoke |
| 1922 | divagr18 | 1.187 | other | legal | unknown | new | JEPA non-record signs-of-life |
| 1923 | jorge-asenjo | 1.06577 | clean | likely-legal | 1855 + modded-nanogpt #181 | B | Asymmetric logit rescale (two learnable softcap scalars trained in TTT) — **regresses** vs #1855 |
| 1925 | simon-marcus | **1.06109** | clean | likely-legal | 1855 | A | MATRIX_LR 0.026→0.028 + PHASED_TTT_PREFIX_DOCS=3500 — pure hyperparam tweaks |
| 1926 | bigbag | 1.06844 | clean | likely-legal | 1874 | B | env-var unlock of #1874 settings; above baseline |
| 1927 | squ11z1 | n/a | other | unknown | 1901/1797/1855 | new | non-record Brotli-11 + stride-2 byte-shuffle compression (artifact-size lever) |
| 1929 | davie2009kh | 0.94569 | prequant-ttt-disputed | likely-illegal | 1738/1229 | C | Scored-Position SLOT (banned per #1647) + pre-quant TTT — double-illegal |
| 1930 | CarlosItp | 1.260 | other | legal | unknown | unknown | non-record QAT int8 RTX 4090 |
| 1931 | jaydenpiao | 1.07586 | clean | likely-legal | 1812 | C | size-clearing fallback (final-block KV int5 + Brotli lgwin=24); not SOTA |

## Headline

**Two independent #1855-base AWQ-style mixed-bit GPTQ submissions claim ~1.0608 BPB.** PR #1908 (romeerp, 1.06081) and PR #1918 (aquariouseworkman, 1.06086) both pick the top-1 most-salient 64-col group per layer and bump it from int6 to int8 inside the existing GPTQ Hessian solve. Independently authored, identical pattern, deltas within seed std of each other and of the #1855 base — but jointly suggest the lever is real (~−0.0003 BPB).

PR #1908 already specced as `research/ideas/1908-awq-lite-mixed-bit-gptq.md`. PR #1918 is the same lever class — treat as confirmation, not a separate spec.

**One actionable hyperparam-only PR:** #1925 (simon-marcus, 1.06109) tweaks `MATRIX_LR=0.028` (from 0.026) and `PHASED_TTT_PREFIX_DOCS=3500` on the #1855 stack and lands below #1797's 1.06157. No code change. Cheap to test on top of our baseline.

**One alternate TTT discipline worth tracking:** #1915 (AidenGeunGeun, 1.06504) — per-document score-first LoRA TTT with no global SGD, per-doc LoRA reset, physical length bucketing. Sits below our baseline but barely. Useful contrast point if PPM-D / pre-quant TTT classes are ruled out and we need a "legal-only" TTT recipe.

**One negative result we can rule out:** #1923 (jorge-asenjo) — Asymmetric Logit Rescale on the #1855 stack regresses by ~0.005 BPB despite identity init. Don't try this.

**One unverified frontier claim to watch:** #1919 (dev-pratap-singh, 1.0587) — author explicitly notes "NOT YET VERIFIED ON H100." If reproduced, becomes the new likely-legal frontier; otherwise discard.

## Closed since last scan

None observed.

## Current leaderboard by category (post-this-scan)

| Category | Best PR | claimed_bpb | verdict |
|---|---|---|---|
| Clean (likely-legal, multi-seed) | #1908 / #1918 | 1.06081 / 1.06086 | clean — AWQ-lite |
| Clean, hyperparam-only | #1925 | 1.06109 | clean — MATRIX_LR + TTT prefix |
| Clean (prior best, single-author 3-seed) | #1855 | 1.06108 | clean — LQER + BOS-fixed SmearGate |
| Validity-pending | #1797 / #1906 | 1.06157 / 1.06136 | flagged by cocohearts |
| PPM-D byte-mixture (suspended) | #1850 / #1854 / #1873 / #1881 | 0.82–1.00 | procedurally suspended pending Issue #1872 |
| SLOT / pre-quant-TTT (banned) | #1929 / #1873 | 0.94 / 0.82 | banned mechanisms |

## Novel levers surfaced (new idea files / notes)

- **#1925 hyperparam tweak** — new idea file: `research/ideas/1925-matrix-lr-ttt-prefix-tune.md`
- **#1915 per-doc LoRA TTT discipline** — new idea file: `research/ideas/1915-per-doc-lora-ttt.md`
- **#1913 multi-temperature AR GPTQ calibration** — noted but not specced (existing `ar-selfgen-gptq-calib.md` covers the AR-self direction; multi-temp is a small variant we can layer later)
- **#1918** — confirmatory evidence for #1908 idea, no separate file

## Map update note

`research/frontier-map.md` is dated 2026-04-20 and predates the entire #1797/#1855/#1855-descendant frontier shift. A full rebuild is overdue. This scan only bumps the snapshot date and adds a "post-2026-04-20 frontier" addendum noting the current leaders and lineage; ASCII tree expansion is deferred — most of the new PRs (#1908/#1918/#1925/#1915/#1923) all descend from #1855 which itself isn't in the map. Flagged as a separate task in `research/ideas/`.
