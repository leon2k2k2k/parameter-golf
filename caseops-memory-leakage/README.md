# CaseOps records — train/val data-leakage audit

**Date:** 2026-05-02. **Working set:** 34 CaseOps-lineage record-track PRs since 2026-04-18 (the merged-record + unmerged-frontier window of CaseOps).

## Headline

The CaseOps records frontier is split into two islands.

**CLEAN island** (8 records): val docs are NOT in the training set.
- #1729 (1.0678) → #1851 (1.06128) → #1868 (1.06141) — the merged trunk, built on the canonical HF dataset `romeerp/parameter-golf-caseops-v1`.
- Plus #1908, #2019, #2031, #2068 — independent forks that explicitly went HF.
- Plus #2027 — non-CaseOps SP8192, clean by pre-CaseOps lineage.

**LEAK island** (25 records): val docs ARE in the training set (~80% overlap).
- Trunk: #1736 (our research baseline) → #1769 → #1787 → #1797 → #1855 → V21 (#1945) → #1953 / #1967 → #2018 → #2118 (current claimed frontier 1.04350).
- All use `prepare_caseops_data.py --val-docs=10000` default → train docs start at canonical-stream index 10,000 → overlap with the regenerated 50k val (docs 0–49,999) on documents 10,000–49,999.
- 40,000 of the 50,000 val docs (80%) appear in the training set and are partially memorized over ~5 epochs.

**Plus:** 1 inherit-only (#2050, eval-only on frozen #1915), and 1 separate symlink-leak mechanism (#2071, claimed 1.0066 via SP8192-symlinked-to-CaseOps).

## Three signposts

- **Leak introduced**: PR #1736 by @dexhunter (2026-04-19) — first record to invoke `prepare_caseops_data.py` with default `--val-docs=10000`, while still reporting `val_tokens=47,851,520` (= a separately-regenerated 50k-doc val set).
- **Leak fixed**: PR #1851 by @aquariouseworkman (2026-04-27) — switched to the HF dataset (`/dev/shm/pgolf_data`, 39-shard subset of the canonical `romeerp/parameter-golf-caseops-v1` build).
- **Leak re-introduced**: PR #1855 by @codemath3000 (same day as #1851) — rebuilt locally with the default `prepare_caseops_data.py`, propagating the leak forward into the V21 / #2018 / #2118 frontier.

## Code-level evidence

This audit operates from primary sources, not READMEs. Verdicts come from:

1. Each PR's shipped shell scripts (`run.sh`, `setup.sh`) — what `--val-docs` flag is passed (if any), or whether `cached_challenge_fineweb.py` is invoked instead.
2. Each PR's shipped `prepare_caseops_data.py` — verified byte-identical across all 22 PRs that ship it: `SHARD_TOKENS = 10_000_000`, `default=10_000` for `--val-docs`. Universal leak-prone defaults.
3. The `cached_challenge_fineweb.py` source (#1729, #2068) — verified to download bin files directly from `romeerp/parameter-golf-caseops-v1` HF dataset using its `manifest.json`, which has `docs_val=50000, docs_train=8,181,945` (sums match → disjoint partition by construction).
4. PR #2018's `DATASET_AUDIT.md` — gold-standard explicit description of the leak construction; transitive evidence for #1855 via byte-identity claim.
5. Train log hparams blocks — `datasets_dir`, `train_shards`, `val_tokens` confirm but never decide alone.

**Searched all 34 PRs' `.sh` files for `--val-docs` overrides — found NONE.** Every record using `prepare_caseops_data.py` uses the default 10,000.

## What this means for our research

- **Our research baseline (PR #1736) is leaky.** All our specs (008 → 300) are measured on data with the val/train overlap. Internally consistent for Δ measurements vs spec-008 baseline; absolute val_bpb numbers should be discounted vs the merged-leaderboard SOTA.
- **The merged-leaderboard SOTA (PR #1851 / #1868 at 1.06128/1.06141) is clean.** That is the actual current frontier in apples-to-apples terms.
- **The unmerged-frontier numbers below 1.05 are inflated by val memorization.** The 0.018 bpb gap from #1851 (1.06128) to #2118 (1.04350) is split between (a) val memorization, (b) genuine recipe improvements (Gated XSA + LQER top-1 + AWQ-lite + AsymLogit), and (c) eval-time overlays (n-gram tilt, GPTQ_RESERVE_SECONDS=2.0). Spec 301 was designed to measure (b) by running #2118 on clean data.
- **Your three submissions (#1779, #1801, #1885) are LEAK-island** by inheritance from #1736 / #1787 / leaky CaseOps prep workflow. (Not in scope of this audit's seed list, but flagged for context.)

## Files in this audit

- `verdicts.md` — the master table with a row for every record (34 nodes), columns for `datasets_dir`, `train_shards`, `val_tokens`, primary verdict (LEAK/CLEAN/INHERIT), mechanism flag, caveats, and stated parent. Read this first for any specific record.
- `family-tree.md` — ASCII trees showing the lineage with `[C]` / `[L]` annotations and explicit "leak introduced / fixed / re-introduced" edge transitions.
- `evidence/<PR>.md` — per-PR evidence snapshot (planned but skipped in this pass; the citation lines are inline in `verdicts.md`).

## How to read this artifact

- Two records with the **same** verdict and **same** val partition (47,851,520 tokens of canonical-stream docs 0–49,999) can be compared directly.
- Two records with **different** verdicts cannot. A LEAK record's val_bpb is inflated downward by memorization; a CLEAN record's val_bpb reports performance on never-seen docs.
- When citing a CaseOps val_bpb anywhere (specs, evaluations, frontier maps), state which island it came from. In particular: do not compare #2118's 1.04350 against #1851's 1.06128 as if the difference is recipe quality.

## Out of scope (not classified here)

- Pre-CaseOps records (≤ 2026-04-18, val_tokens=40,540,160 from `download_hf_docs_and_tokenize.py NUM_VAL_DOCS=50000`, clean by lineage).
- PPM-D submissions (#1850, #1872, #1885 etc.) — separate legitimacy track.
- Records outside the 34-node working set.
- Eval-time leaks vs train-time leaks: this audit is only about train/val document overlap. The within/word `boundary_lut` C1 leak in #1967 / #2018 lineage is a *separate* code-bug leak that is documented elsewhere and noted in `verdicts.md` for the affected nodes.

## Source-of-truth references

- HF dataset manifest (canonical clean reference): https://huggingface.co/datasets/romeerp/parameter-golf-caseops-v1/raw/main/datasets/manifest.json
- PR #2018 `DATASET_AUDIT.md` (gold-standard leak description): inside the unmerged PR diff at `records/track_10min_16mb/2026-04-30_GatedXSA_LQERTop1_IntimerNgramTTT/DATASET_AUDIT.md`
- PR #2118 `submission.json` (current frontier admits leak): `technique_summary` field literal text says `--val-docs=10000 train shards + 50k val eval`.

---

*Audit produced 2026-05-02 in response to spec 301 interview surfacing the leak in PR #2118. Methodology: chronologically classify the user-supplied seed list of 31 records + 3 discovered ancestors (#1908, #1923, #2007), processing earliest-first so each child can lean on already-verified parent verdicts. Code-level verification on all 34 nodes via shipped `.sh` / `.py` files. No reliance on `frontier-state.json` as primary evidence.*
