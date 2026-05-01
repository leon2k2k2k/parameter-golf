---
spec: 301
slug: pr2118-tilt-off
date: 2026-05-01
status: frozen
idea: (none — direct ablation, no prior idea file)
---

# Spec 301 — PR #2118 with n-gram tilt OFF

## Hypothesis

PR #2118 (aquariouseworkman, claimed val_bpb **1.04350** / 3-seed) wins by stacking the same leaky token-only n-gram tilt audited in PRs #2018/#1967 (per memory: identical `online_ngram_state.c`, `boundary_lut[tokens[i]]` C1 leak, ~95% of gated mass leaky) on top of legitimately better levers: corrected CaseOps data prep, GPTQ_RESERVE_SECONDS=2.0, plus the unchanged Gated XSA / LQER top-1 / AWQ-lite / AsymLogit lineage from #2018.

Disabling the tilt isolates the legal portion. The training objective is **base NLL** — tilt is explicitly a scoring-time overlay (`train_gpt.py:3764`), so the same training run + EMA + quant + phased TTT applies.

## Baseline

- **PR #2118 tilted final** (3-seed): 1.04350
- **PR #2118 seed 42 diagnostic stages** (from `train_seed42.log`):
  - pre-quant post-EMA: 1.04683
  - quantized (no TTT, no tilt): 1.05547
  - quantized + TTT + **tilt**: 1.04295
- **PR #2018 quantized (no TTT, no tilt) 3-seed mean**: ~1.0588
- **Frontier comparable**: PR #1851 BOS-fix at 1.06128

## Expected Δ

Stripping tilt should land in **1.048 – 1.055** (point estimate 1.052), based on:
- The 0.01252 gap between seed-42's `diagnostic_quantized` (1.05547, no TTT, no tilt) and `quantized_ttt_phased` (1.04295, TTT + tilt) splits between TTT (real) and tilt (mostly leaky).
- TTT alone on PR #2018 lineage moves quant→final by ~0.012; here TTT + tilt moves it by ~0.0125, so tilt's marginal contribution is small (~0.0005) ONLY IF TTT is dominating. More plausibly TTT contributes ~0.005–0.008 and tilt ~0.005–0.007.
- PR #2118's pre-training is ~0.003 better than PR #2018 (corrected CaseOps prep + ~68 more steps), so any number 1.048–1.055 is +0.006 to +0.013 ahead of the 1.06128 frontier.

Confidence the legal final lands ≤ 1.060: ~80%. Confidence ≤ 1.055: ~55%.

## Accept criteria

- Seed 1 final `quantized_ttt_phased val_bpb`:
  - **≤ 1.055** → fire seeds 2 + 3, this is a record-track candidate.
  - **1.055 – 1.060** → marginal; pause and decide with user.
  - **> 1.060** → kill, do not run remaining seeds.
- SOTA std reference is ~0.0002; a single seed can't decide alone, but a clear win/loss vs 1.060 is informative at one seed.

## Config diff (vs PR #2118 default submission)

Single environment override at launch:

```
NGRAM_TILT_ENABLED=0
```

Everything else identical to the PR #2118 submission environment. In particular:
- `PHASED_TTT_ENABLED=3` (per memory: "=0 means slow TTT, always use =3"), `PHASED_TTT_NUM_PHASES=3` — keep whatever PR #2118 uses; do not override.
- `NGRAM_HINT_PRECOMPUTE_OUTSIDE` — irrelevant when tilt is off.
- `GPTQ_RESERVE_SECONDS=2.0` — keep PR #2118 default.

## Code changes

**No edits.** Pin to PR #2118 commit:

- Repo: `aquariouseworkman/parameter-golf` (PR #2118 fork)
- Branch: PR #2118 head
- Commit: `30a3d90013af9962e677f754dcd1bf504a475dba`
- Submission dir: `records/track_10min_16mb/2026-05-01_Gated XSA + token-only n-gram TTT + GPTQ_RESERVE=2.0/`

Tilt path is gated by `ngram_tilt_enabled` at all 4 call sites (`train_gpt.py:387`, `:3503`, `:3559`, `:4307`). The flag defaults to "0" via `os.environ.get("NGRAM_TILT_ENABLED", "0")`, so just don't export `NGRAM_TILT_ENABLED=1` (or set it to 0 explicitly).

**No `exp/<slug>` branch needed** — env override on a tested upstream commit.

## Hardware ladder

**Skip mini.** This is an env-flag-only change on a 3-seed-validated upstream commit; no code touched. Going straight to 8×H100 with staged seed plan.

- **Stage 1 (seed 42 only):** 8×H100 SXM 80GB, ~10 min train + ~10 min eval ≈ ~$4.
- **Stage 2 (seeds 1234 + 314):** Only if Stage 1 passes accept criteria. ~$8.

## Seed plan

Use PR #2118's exact seeds for direct comparability: **42, 1234, 314**.

Stage 1 = seed 42 alone. Stage 2 = seeds 1234 + 314 in parallel on the same pod (or sequentially if capacity tight).

## Inputs

PR #2118 introduces a "corrected" CaseOps data preparation (`prepare_caseops_data.py`, 177 LOC, `--val-docs=10000`). **Critical preflight question** — do we already have the corrected shards on the volume?

- **Tokenizer**: `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — shipped in the PR; copy into pod or use one already on volume.
- **Data layout** (per `prepare_caseops_data.py` docstring):
  ```
  ./data/datasets/fineweb10B_sp8192_caseops/datasets/
    fineweb_{train,val}_XXXXXX.bin
    fineweb_val_bytes_XXXXXX.bin
  ```
- **If volume already has CaseOps shards but not the "corrected" 10k-val variant**: regenerate via `prepare_caseops_data.py` from raw `docs_selected.jsonl`. CPU-only, runs once.
- **Hotstart checkpoint**: PR #2118 trains from scratch (5000 steps, ~10 min) — no hotstart needed.

## Checkpoints to emit

Standard PR #2118 outputs only:
- `final_model.pt` (post-EMA, pre-quant) — eval as `prequant_val_bpb`.
- `final_model.int6.ptz` (quantized) — submission artifact.
- `train_seed{42,1234,314}.log` — full training + eval log.
- `submission.json` — auto-generated.

No extra retention; no intermediate checkpoint snapshots needed.

## Stop-early criteria

- NaN at any step → stop, report.
- val_bpb at step 4000 > 1.10 → divergence, stop.
- Step time > 130ms (sustained) → throughput tax > expected, investigate before continuing.
- **Mid-run torch.compile recompile** → kill immediately (per memory). Should not occur: this is the unmodified PR #2118 commit, prewarmed cache from a prior PR #2118 submission would help if available.

## Cost estimate

- Stage 1 (1 seed, 8×H100 SXM): ~$4 train + eval, ~20–25 min wallclock.
- Stage 2 (2 seeds): ~$8, can run sequentially on one pod (~50 min) or in parallel on two pods.
- Total worst-case: **~$12–15**.

## Extra artifacts

Beyond EXECUTION.md defaults:
- Capture the full final eval log block (search for `quantized_ttt_phased val_loss:`).
- Record the `prequant_val_bpb` and `diagnostic_quantized val_bpb` lines — needed to decompose tilt vs TTT contribution.

## Open questions for execution interview

1. **Corrected CaseOps shards on volume**: do we have `data/datasets/fineweb10B_sp8192_caseops/datasets/` with the 10k-val variant on either NE-1 or JP volume? If not, regenerate via `prepare_caseops_data.py` (needs `docs_selected.jsonl`). If `docs_selected.jsonl` is also absent, halt and ask before downloading FineWeb-10B fresh.
2. **Launch invocation**: PR #2118 ships no `run_*.sh`. Confirm with user how the author launches (probably the standard `torchrun` harness with a small wrapper); we'll assemble launcher mirroring the env block from `train_seed42.log:1-90` with `NGRAM_TILT_ENABLED=0` instead of `=1`.
3. **Prewarm**: do we have an inductor cache from any prior PR #2118-lineage run? If not, factor in ~5–10 min one-time prewarm (per memory). If we do, restore via `tmp_exec/cache_restore.sh`.
4. **Region**: NE-1 first, JP fallback (per memory).
5. **Failure mode**: if `NGRAM_TILT_ENABLED=0` doesn't cleanly bypass (e.g. some path still imports `online_ngram_tilt` and crashes), **halt and ask** — do not patch around it on the pod.
6. **Monitoring cadence**: 1-min cron polling, multi-arm comparison table (seed 42 vs PR #2118 reference points: step 4000 ≈ 1.090, step 5002 ≈ 1.054 untilted-base / 1.043 tilted-final-equivalent).

## Provenance

- PR #2118: https://github.com/openai/parameter-golf/pull/2118
- Memory: `project_2018_1967_c1_leak.md` documents the boundary_lut leak in the shared `online_ngram_state.c`.
- Memory: `project_baseline_1851_post_1872.md` confirms PPM-D class suspended pending #1872 — same family of concern as the n-gram tilt here.
