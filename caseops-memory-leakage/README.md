# CaseOps records — train/val data-leakage audit

**Date:** 2026-05-02 (with strict re-audit applied same day). **Working set:** 34 CaseOps-lineage record-track PRs since 2026-04-18 (the merged-record + unmerged-frontier window of CaseOps).

## Headline (after strict re-audit)

The CaseOps records frontier is split into clean / leak / ambiguous islands.

**CLEAN island** (9 records, val docs NOT in training set):
- Merged trunk: #1729 (1.0678) → #1851 (1.06128) → #1868 (1.06141) — built on canonical HF dataset `romeerp/parameter-golf-caseops-v1`.
- Independent HF forks: #1908 (1.06081), #2019 (1.05847), #2031 (1.05985), #2068 (1.06172).
- **#1945 (1.05943)** — flipped from LEAK to CLEAN in the re-audit; `finalize_v18.sh` reveals snapshot_download from HF.
- #2027 — non-CaseOps SP8192, clean by pre-CaseOps lineage.

**LEAK island** (21 records, val docs ARE in training set, ~80% overlap):
- Trunk: #1736 (our research baseline) → #1769 → #1787 → #1797 → #1855 → V21 / #1923 / #1967 → #2018 → #2118 (current claimed frontier 1.04350).
- All these have direct evidence: explicit `prepare_caseops_data.py` invocation, audit/submission.json admission, or train log path with `_caseops/datasets/datasets/<name>` triple-nesting / single `<root>/datasets/<name>` (which only local prep produces — HF always gives double-nesting).
- All use `--val-docs=10000` default → train starts at canonical-stream doc 10,000 → overlap with 50k val (docs 0–49,999) on documents 10,000–49,999.

**AMBIGUOUS** (3 records, cannot resolve from PR artifacts alone):
- **#1953** (1.05855), **#2041** (1.05692), **#2075** (no claim) — no explicit prep evidence in their PR. Path matches HF target. Initial first-pass LEAK calls were over-confident (relied on lineage-inheritance heuristic). Lean CLEAN but require external evidence (commit logs from author's pod, etc.) to confirm.

**INHERIT** (1): #2050 — eval-only on frozen #1915 quantized artifacts.

**Separate mechanism** (1): #2071 — claimed 1.0066 via SP8192-symlinked-to-CaseOps; orthogonal to val10k-train, audit-flagged.

## Honest current frontier

If the AMBIGUOUS records are CLEAN (path + parent verdict makes this likely):
- Best clean BPB: **#2019 at 1.05847** (@aquariouseworkman, unmerged), then #1953 at 1.05855 (if AMBIGUOUS resolves CLEAN), #1945 at 1.05943, #2031 at 1.05985, #1908 at 1.06081, #1851 at 1.06128 (merged).

If the AMBIGUOUS records are LEAK:
- Best clean BPB: #2019 at 1.05847, then #1945 at 1.05943, then #2031 at 1.05985, then #1908 at 1.06081, then merged #1851 at 1.06128.

Either way, the current claimed frontier #2118 at 1.04350 is **definitely LEAK**, and the clean frontier is **at most ~0.012 bpb below #2118 (i.e., realistic clean SOTA is ≥ 1.05847).**

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
