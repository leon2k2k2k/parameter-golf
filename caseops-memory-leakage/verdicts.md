# CaseOps records — train/val data-leakage verdicts

**As of 2026-05-02.** Every CaseOps-lineage record (merged + unmerged) since 2026-04-18.

**Working set:** 34 records (31 from user's seed list + 3 ancestors discovered: #1908, #1923, #2007).
**Boundary nodes (no per-classification):** #1493, #1626 (pre-CaseOps lineage, clean by `download_hf_docs_and_tokenize.py NUM_VAL_DOCS=50000`).

## Tally

| Verdict | Count | Records |
|---|---:|---|
| **CLEAN** | 9 | #1729, #1851, #1868, #1908, **#1945** *(re-audit: finalize_v18.sh has snapshot_download)*, #2019, #2027 (non-CaseOps), #2031, #2068 |
| **LEAK** | 20 | #1736, #1769, #1787, #1797, #1855, #1923, #1967, #2007, #2018, #2060, #2071 (symlink), #2078, #2100, #2101, #2109 (custom variant), #2117, #2118, #2121, #2123, #2124 |
| **AMBIGUOUS** | 4 | **#1953, #2014, #2041, #2075** *(re-audit r2: no direct prep evidence in PR; path could match either HF or local-prep; lean direction noted per row)* |
| **INHERIT** | 1 | #2050 (eval-only on #1915) |

> **Re-audit applied 2026-05-02** (after first pass). Strict criterion: a LEAK verdict requires at least one of (a) explicit shell-script invocation of `prepare_caseops_data.py` without `--val-docs=50000`, (b) README "Data setup" matching the actual train log path, (c) audit/`submission.json` admission text, (d) train-log path with `_caseops/datasets/datasets/<name>` triple-nesting OR single `<root>/datasets/<name>` (which only `prepare_caseops_data.py` produces — HF always gives double-nesting). Records that previously got LEAK by lineage-inheritance alone are now AMBIGUOUS unless they meet one of these tests.

## What "LEAK" means

For records flagged `val10k-train+50k-val-regen` (the dominant leak class):
- Train shards built from `prepare_caseops_data.py` with default `--val-docs=10000` → train documents start at canonical-stream index **10,000**.
- Val files contain the first **50,000** canonical-stream documents (`val_tokens: 47,851,520`).
- → Documents at canonical-stream indices **10,000–49,999 (40,000 docs)** appear in both train and val.
- During training, the model sees those 40,000 docs ~5 epochs and partially memorizes them; they are then scored as if held out.

Other leak mechanisms surfaced separately: `symlink-leak` (#2071, audit-flagged), `custom-variant` (#2109, MP3 marker-pair fusion alters val partition).

## What "CLEAN" means

Records flagged `hf-dataset`:
- Train + val sourced from the canonical HF dataset `romeerp/parameter-golf-caseops-v1`, which has a strict `--val-docs=50000` partition baked into its `manifest.json` (`docs_val=50000, docs_train=8,181,945, docs_total=8,231,945`, sums match exactly).
- Val docs (0–49,999) and train docs (50,000+) are disjoint by construction.

## Verdict signal authority (CODE-LEVEL)

In order of strength (highest first):
1. **Code in shipped scripts**: `setup.sh` / `run.sh` invocations of `prepare_caseops_data.py` (look at `--val-docs` flag) or `cached_challenge_fineweb.py` (look at `--variant`).
2. **`prepare_caseops_data.py` constants** in each PR: every shipped copy has byte-identical `SHARD_TOKENS = 10_000_000` and `default=10_000` for `--val-docs`. (Verified across all 22 PRs that ship it.)
3. **`cached_challenge_fineweb.py` logic**: downloads bin files directly from `romeerp/parameter-golf-caseops-v1` HF dataset using its manifest's `files_train`/`files_val` counts. The HF manifest pins `docs_val=50000, docs_train=8,181,945, sums match` → CLEAN by construction.
4. **Audit/manifest text** — `DATASET_AUDIT.md` (PR #2018), HF `manifest.json`, `submission.json.technique_summary`.
5. **Train log** `datasets_dir`, `train_shards`, `val_tokens` lines from the actual run — confirms but doesn't decide alone.
6. **README data-prep section** — corroborating; READMEs can be misleading (e.g. PR #2118 titled "corrected CaseOps data preparation" but its `submission.json` admits `--val-docs=10000`).

`frontier-state.json` was NOT used as primary evidence; verdicts here are regenerated from primary sources.

### Code-level binary distinction

- **PR ships / invokes `cached_challenge_fineweb.py`** (downloads from `romeerp/parameter-golf-caseops-v1` HF dataset) → **CLEAN**. The HF dataset is provably built with `--val-docs=50000` partition.
- **PR ships / invokes `prepare_caseops_data.py` without `--val-docs=50000` override** → **LEAK**. Default is 10,000. We searched all `.sh` files in all 34 PRs; **NO PR overrides `--val-docs` to a non-default value**. Every prepare-based record uses the default 10,000.

## Master table

| PR | Author | Date | val_bpb | Stated parent | datasets_dir | train_shards | val_tokens | **Verdict** | Mechanism | Caveats / evidence |
|---|---|---|---:|---|---|---:|---:|---|---|---|
| **#1729** | @romeerp | 2026-04-19 | 1.0678 | #1626 | `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | 80 | 47,851,520 | **CLEAN** | hf-dataset | First CaseOps record. README explicitly: "downloaded from `romeerp/parameter-golf-caseops-v1` with `cached_challenge_fineweb.py`". |
| **#1736** | @dexhunter | 2026-04-19 | 1.06549 | #1729 | `./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **LEAK INTRODUCED HERE.** README invokes `prepare_caseops_data.py --docs ./fineweb10B_raw/docs_selected.jsonl ...` with no `--val-docs` flag → defaults to 10,000. Currently OUR research baseline. |
| **#1769** | @dexhunter | 2026-04-22 | 1.06453 | #1736 | same triple-nested local prep | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Same prep as #1736. |
| **#1787** | @nprime06 | 2026-04-23 | 1.06335 | #1736, #1769 | `/workspace/src/parameter-golf/data/datasets/...caseops/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Same prep workflow. |
| **#1797** | @dexhunter | 2026-04-25 | 1.06157 | #1787 | local triple-nested | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | SmearGate + LQER on #1787; same prep. |
| **#1851** | @aquariouseworkman | 2026-04-27 | **1.06128** | #1787 (via #1797) | `/dev/shm/pgolf_data` | **39** | 47,851,520 | **CLEAN** | hf-dataset | **LEAK FIXED HERE.** Switched to HF dataset (39-shard subset in RAM). Current merged-SOTA leader. |
| **#1855** | @codemath3000 | 2026-04-27 | 1.06108 | #1787, #1797 | `/workspace/pr1797_work/data/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **LEAK RE-INTRODUCED HERE.** Author rebuilt locally with `prepare_caseops_data.py` default. PR #2018's `DATASET_AUDIT.md` verifies byte-identity with `--val-docs=10000` default. The 04-27 1.0611 record. |
| **#1868** | @Christopher-Lee-McClendon | 2026-04-29 | 1.06141 | #1851 | `/dev/shm/pgolf_data` | 39 | 47,851,520 | **CLEAN** | hf-dataset | 3-seed reproduction of #1851's BOS-fix on HF dataset. |
| **#1908** | @romeerp | 2026-04-28 | 1.06081 | #1855 | `/workspace/parameter-golf-pr1855-clean/data/datasets/...v1_reserved` | ? | ? | **CLEAN** | hf-dataset | README explicit: "sourced from Hugging Face: `romeerp/parameter-golf-caseops-v1`". |
| **#1923** | @jorge-asenjo | 2026-04-29 | 1.05971 | #1855 | `/workspace/pg-data/datasets/...v1_reserved` | 1502 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Code-level: README admits original `val_tokens=9,662,464` (= stock `prepare_caseops_data.py --val-docs=10000` 1-shard output, docs 0–9,999). Re-pulled ONLY the val file from HF after corruption → val now docs 0–49,999. Train was never replaced → still docs 10,000+ from default prep. Overlap docs 10,000–49,999. (1502 train shards from a custom prep on a larger docs file; doesn't change the partition mechanism.) |
| **#1945** | @alertcat | 2026-04-29 | 1.05943 | #1855, #1908, #1923 | `/workspace/caseops_data/datasets/datasets/...v1_reserved` | 80 | 47,852,288 | **CLEAN** | hf-dataset | "V21" = #1855 + AWQ-lite + AsymLogit. **Re-audit:** `finalize_v18.sh` contains `snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', local_dir='/workspace/caseops_data')`. Train log path matches HF target exactly. README's `prepare_caseops_data.py` "Data setup" section is stale documentation — actual run used HF. (val_tokens off by 768 from canonical 47,851,520 = alignment artifact, identical val partition.) |
| **#1953** | @andrewbaggio1 | 2026-04-30 | 1.05855 | #1945 | `/workspace/caseops_data/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | **Re-audit downgrade:** PR ships only `train_gpt.py` + logs, no setup/finalize script. README has no data prep instructions (only credit lines to PR #1908, #1736, #1729). Path matches HF target exactly (same as #2019 and #1945, both confirmed CLEAN). Parent #1945 is CLEAN. **Lean CLEAN** but no direct evidence in PR alone. |
| **#1967** | @ndokutovich | 2026-04-30 | 1.05851 | #1945 | `/runpod-volume/caseops_data/datasets/datasets/...v1_reserved` | 1499 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Code-level: ships `setup.sh` invoking `python3 prepare_caseops_data.py --docs $DOCS_JSONL --out $DATA_DIR --sp ...` with NO `--val-docs` flag → default 10,000. (1499 train shards from larger docs file; partition mechanism unchanged: train starts at doc 10,000, overlaps 50k val on docs 10,000–49,999.) Also has within/word `boundary_lut[tokens[i]]` C1 leak — separate code-bug leak. |
| **#2007** | @Elubrazione | 2026-04-30 | 1.05899 | #1855 (technique-implicit) | `/root/blockdata/pg-data/datasets/...caseops/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | LongCtx + NoQV. Triple nesting + ships `prepare_caseops_data.py`. |
| **#2014** | @simonbissonnette | 2026-04-30 | 1.05759 | #1855, #1953 | `/dev/shm/pgolf_caseops_data_80_l17_final` | ? | ? | **AMBIGUOUS** | hf-or-local-prep | **Re-audit (round 2):** README implies "uses same shards as PR #1855" (LEAK by inheritance) but does NOT explicitly invoke prep — only references `prepare_caseops_data.py` as fallback "if you don't have them". No HF download script, no setup.sh. Path is custom RAM-disk flat (not triple-nested). Could be either HF-flattened (via `cached_challenge_fineweb.py`-style download) or local-prep copy. **Lean LEAK** because the "same shards as #1855" English implies literal data inheritance, but no direct evidence. |
| **#2018** | Simon Marcus | 2026-04-30 | 1.04722 | #1945, #1967, #1953, #1855 | `/tmp/pr1855_compact_train_full50k_val/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **GOLD-STANDARD LEAK DOC.** `DATASET_AUDIT.md` explicitly states `--val-docs=10000` train + 50k val regen + first 80 train shards verified byte-for-byte. |
| **#2019** | @aquariouseworkman | 2026-04-30 | 1.05847 | #1855 | `/workspace/caseops_data/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **CLEAN** | hf-dataset | README explicit: `snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1')`. Path matches HF layout. |
| #2027 | @H1cSuNtDr4C0n3S | 2026-04-30 | 1.08064 | #1493 | `/workspace/parameter-golf-qrescue-20260426/data/datasets/fineweb10B_sp8192` | ? | ? | **CLEAN** | pre-caseops-pipeline | Non-CaseOps SP8192 lineage; out-of-CaseOps-scope. Clean by lineage. |
| **#2031** | @deborahnelson8788726 | 2026-04-30 | 1.05985 | #1855 | `/workspace/parameter-golf-final/romeerp_caseops_first39/datasets/datasets/...v1_reserved` | 39 | 47,851,520 | **CLEAN** | hf-dataset | README explicit: "canonical romeerp/parameter-golf-caseops-v1 shards with 39 train shards, one validation token shard". |
| **#2041** | @jorge-asenjo | 2026-04-30 | 1.05692 | #1945, #1967, #2018 | `/workspace/pg-data/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **AMBIGUOUS** | hf-or-local-prep | **Re-audit downgrade:** No prep invocation in README; no shell script in PR. Path is double-nested (no `_caseops/` middle), consistent with EITHER HF snapshot_download to `/workspace/pg-data` OR `prepare_caseops_data.py --out=/workspace/pg-data/datasets`. Same author (@jorge-asenjo) as confirmed-LEAK #1923, but #1923's evidence was its own README admission, not a shared workflow. |
| #2050 | @someone114514 | 2026-04-30 | 1.06083 | #1915 | `./data/datasets/...v1_reserved` | ? | 47,851,520 | **INHERIT** | inherit-from-#1915 | Eval-only on frozen #1915 quantized artifacts. Data verdict depends on #1915 (not in working set). |
| **#2060** | @S0urC10ud | 2026-04-30 | 1.05792 | #2007 | `/root/blockdata/pg-data/datasets/...caseops/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | 5-knob retune of #2007 (LEAK). |
| **#2068** | @jayaram1125 | 2026-04-30 | 1.06172 | #1797 | `./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **CLEAN** | hf-dataset | README explicit: `cached_challenge_fineweb.py --variant sp8192_lossless_caps_caseops_v1_reserved --train-shards 80` from romeerp HF. Stages into leaky-looking dir post-download (path is misleading). |
| **#2071** | @jamesEmerson112 | 2026-04-30 | **1.0066** (claimed) | #1851 | `./data/datasets/fineweb10B_sp8192` | ? | ? | **LEAK** | symlink-leak | **DIFFERENT LEAK MECHANISM.** Audit-flagged: `caseops_enabled=False` env, but pod data paths symlinked to CaseOps-tokenized shards. README admits: "active via symlinked data". Orthogonal to val10k-train. |
| **#2075** | @deusexnatura | 2026-04-30 | (no claim) | #1855 | `/workspace/caseops_data/datasets/datasets/...v1_reserved` | ? | ? | **AMBIGUOUS** | hf-or-local-prep | **Re-audit downgrade:** Ships `prepare_caseops_data.py` as a shipped file but README does NOT explicitly invoke it. Path matches HF target. No setup.sh. AMBIGUOUS. |
| **#2078** | @hi-aduek | 2026-04-30 | 1.05804 | #2014 | `/dev/shm/caseops1851-data/datasets/...caseops/datasets/datasets/...v1_reserved` | ? | ? | **LEAK** | val10k-train+50k-val-regen | #2014 reproduction; triple nesting under /dev/shm; same prep workflow. |
| **#2100** | @someone114514 | 2026-04-30 | 1.05807 | #2060 | `/root/blockdata/pg-data/datasets/...caseops/datasets/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | LongCtx + No-QV + Prefix3500. Same lineage prep. |
| **#2101** | @OnlyJundong | 2026-05-01 | 1.05845 | #1855 | `/workspace/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | AWQ-lite + AsymLogit + GradCentral. Ships `prepare_caseops_data.py` (default). |
| **#2109** | @izlley | 2026-05-01 | 1.05917 | #1855 | `/workspace/data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3` | 1497 | **36,562,944** | **LEAK** | custom-variant | MP3 marker-pair fusion. **CUSTOM dataset variant** (different val_tokens, ~10M-token shards differ). Ships `prepare_caseops_data.py` and `prepare_marker_pair_v3.py`. Inherits same val10k-train mechanism but with vocab surgery. |
| **#2117** | @JulianTang2027 | 2026-05-01 | (3-seed reproduction) | #2101 | `./data/datasets/...v1_reserved` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | 3-seed reproduction of #2101 (LEAK). |
| **#2118** | @aquariouseworkman | 2026-05-01 | **1.04350** | #2018 | `/workspace/data_correct` | 80 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | **CURRENT FRONTIER (claimed).** `submission.json.technique_summary` literal text: `"--val-docs=10000 train shards + 50k val eval"`. Same author who shipped clean #1851 a week earlier. |
| **#2121** | @Kbediako | 2026-05-01 | 1.06099 | #1855 | `/workspace/pg_stageb_v2_seed0_1234/datasets/...v1_reserved` | ? | ? | **LEAK** | val10k-train+50k-val-regen | StageB v2 CaseOps TTT. Ships `prepare_caseops_data.py`. |
| **#2123** | @vaibhavmishra1 | 2026-05-01 | 1.05933 | #1855 | `./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/...v1_reserved` | 78 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Closed; superseded by #2124. |
| **#2124** | @vaibhavmishra1 | 2026-05-01 | 1.05933 | #1855 | same | 78 | 47,851,520 | **LEAK** | val10k-train+50k-val-regen | Resubmission of #2123. |

## Code-level resolution of #1923 and #1967

Initially marked AMBIGUOUS because their `train_shards` counts (1502, 1499) didn't match HF (80 × 100M) or standard prep (~800 × 10M). Code-level inspection resolved both:

- **#1967**: `records/track_10min_16mb/2026-04-30_NgramTilt_V21_LeakyReLU_1.05851/setup.sh` directly invokes:
  ```bash
  python3 "$(dirname "$0")/prepare_caseops_data.py" \
      --docs "$DOCS_JSONL" \
      --out "$DATA_DIR" \
      --sp "..."
  ```
  No `--val-docs` flag → defaults to 10,000. Confirmed LEAK.

- **#1923**: README admits the original val_tokens was `9,662,464` (the canonical stock output of `prepare_caseops_data.py --val-docs=10000` on the canonical docs stream — 1 val shard, docs 0–9,999). After review, only the val file was re-pulled from HF (val now docs 0–49,999). Train shards were never replaced → still docs 10,000+. Overlap on docs 10,000–49,999 = LEAK.

The unusual 1499 / 1502 train_shards counts are explained by these PRs running `prepare_caseops_data.py` on a larger `docs_selected.jsonl` than the canonical 8.23M-doc version — the partition logic is the same, just with more total shards generated. The val/train overlap mechanism is unchanged.

### #2071 — separate `symlink-leak` mechanism

PR #2071 claims val_bpb=1.0066 by setting `caseops_enabled=False` while symlinking pod data paths to CaseOps-tokenized shards. The training run reads CaseOps shards but reports stats as if SP8192. This is orthogonal to the val/train overlap mechanism we audit here — it's flagged separately as `symlink-leak` and called out in the audit context surrounding the PR.

### #2109 — custom MP3 dataset variant

PR #2109 fuses `[▁, MARKER]` 2-grams via `prepare_marker_pair_v3.py`, producing a custom `fineweb10B_sp8192_caseops_marker_pair_v3` dataset variant. val_tokens differs (36,562,944) from the canonical 47,851,520 because of vocab surgery on the val side. Underlying canonical-stream partition still inherits from the same val10k-train workflow.

### #2027 and #2071 — non-CaseOps lineage

Both records are in the date window but use SP8192 (non-CaseOps) data — they don't share the val10k-train+50k-val-regen mechanism. #2027 is clean by pre-CaseOps lineage. #2071 has an orthogonal symlink leak.

## How to interpret val_bpb across this table

Two records with **different** verdicts cannot have their val_bpb directly compared:
- A LEAK record's val_bpb is inflated downward because the model has memorized 80% of its val docs during training.
- A CLEAN record's val_bpb reports performance on docs the model never saw.

Two records with the **same** verdict and **same** val partition (47,851,520 tokens of canonical-stream docs 0–49,999) can be compared directly against each other.

The 0.018 bpb gap between the LEAK frontier (#2118 at 1.04350) and the CLEAN frontier (#1851/#1868 at 1.06128/1.06141) is the combined effect of:
1. Memorization of ~40,000 val docs (data leak component, ~0.005–0.012 bpb estimated)
2. Genuine recipe improvements (Gated XSA, LQER top-1, AWQ-lite, AsymLogit, etc., ~0.000–0.013 bpb)
3. Eval-time overlays (n-gram tilt, ~0.005–0.012 bpb, separate from pretraining)

Distinguishing (1) from (2) requires running #2118's recipe on the clean HF dataset, which is what spec 301 was designed to do.
