# Spec 052 — PPM-D port to 047B with tuned anti-hijack gating

**Date:** 2026-04-28
**Branch:** `exp/052-ppm-port-tuned` @ TBD (commit pending push)
**Idea:** `research/ideas/ppm-port-on-047B.md` (synthesized from `eval/2026-04-28_ppm-rigorous-full-val.md`)
**Phase 1 validated:** measured `mix_bpb_sidecar = 1.00506` on 1×H100 EVAL_ONLY against 047B's `final_model.int6.ptz` (this very pipeline, end-to-end). See `runs/052-phase1-eval/eval_ppm4.log`.

## Hypothesis

PR #1850's PPM-D byte mixture, ported onto our 047B post-quant model, with two tuned hyperparameters from a 2D sweep on cached per-byte data, achieves a 3-seed mean `val_bpb ≈ 1.005`, matching 1850's submission within noise.

The tuned settings:
- `ppm_conf_threshold = 0.76` (vs 1850's 0.9) — fires the gate more aggressively
- `ppm_nn_skip_thr_nats = 0.277` (= 0.40 bits, NEW in this spec; 0 disables) — anti-hijack guard that suppresses the gate when NN already gave the actual byte > exp(-0.277) ≈ 0.76 probability

The two knobs **interact strongly**: lower `thr` alone gains 0.002 BPB; anti-hijack alone gains 0.007; both together gain 0.026.

## Baseline

047B post-quant published `val_bpb = 1.07647` (reproduced in Phase 1 EVAL_ONLY to 0.00006). PR #1850 submission `val_bpb = 1.00495` (3-seed mean).

## Expected Δ

`mix_bpb_sidecar = 1.005 ± 0.002` across 3 seeds. Targets parity with 1850, NOT a leapfrog. Beating 1850 requires Phase 2 (TTT stack or stronger NN base — out of scope here).

## Accept criteria

- 3-seed mean `mix_bpb_sidecar ≤ 1.008` (would tie 1850 within 1-σ noise band)
- Per-seed `mix_bpb_sidecar ≤ 1.010`
- Total `eval_time` ≤ 600s (verified: Phase 1 took 250s on 1×H100; on 8×H100 should be faster due to parallel forward)
- No legality regression (Issue #1017 conditions 1–4):
  1. Causal — model attention is causal; PPM context is byte-prefix only
  2. Normalized — PPM-D and NN softmax both normalized; mixture in probability space
  3. Score-before-update — PPM tables incremented AFTER each byte's mix is recorded
  4. Single L→R pass — no rescore/selection, single-context PPM (no chunk reset)

## Config diff (vs 047B EVAL_ONLY default)

```
PPM_NATIVE_ENABLED=1            # NEW: enables PPM path (default 0)
PPM_ORDER=4                     # match 1850
PPM_LAMBDA_HI=0.9               # match 1850
PPM_LAMBDA_LO=0.05              # match 1850
PPM_CONF_THRESHOLD=0.76         # tuned (vs 1850's 0.9)
PPM_NN_SKIP_THR_NATS=0.277      # NEW knob (anti-hijack)
PPM_LOG_CACHE_SIZE=1048576
PPM_OMP_THREADS=8
PPM_OMP_CHUNK_TOKENS=0          # 0 = single-pass (1850-style); >0 = OMP chunked (1857-style, ~0.005 BPB worse)
EVAL_ONLY=1                     # no training, score existing 047B blob
SEED ∈ {42, 7, 1337}            # match 1850's seeds for direct comparability
```

All other env vars are seeded from `runs/047B-loop-kv-shrink-screen/train.log` via the wrapper, then `DISTRIBUTED RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT` are unset for 1×H100 single-process, OR set via torchrun for 8×H100.

## Code changes

Branch `exp/052-ppm-port-tuned` adds, on top of 047B's `train_gpt.py`:

1. **PPM C source** (~131 lines, embedded as `_NATIVE_PPM_C_SRC` r-string) — port of 1850's `score_byte` / `ppm_score` / `ppm_score_omp` plus our anti-hijack guard:

   ```c
   // 5-line patch in score_byte:
   int hi_raw = (conf >= thr);
   int hi = hi_raw && !(nn_skip_thr > 0.0 && nn_logp > -nn_skip_thr);
   double lam = hi ? lambda_lo : lambda_hi;
   (*gate_total)++;
   if (hi) (*gate_high)++;
   ```

2. **Python helpers** — `_build_native_ppm_lib()`, `_build_token_bytes_lut_for_ppm()`, `_ppm_mixture_bpb_native()`. Faithful port of 1850's wrappers, plus `nn_skip_thr` argument plumbing.

3. **`run_ppm_native_pass()`** — collects per-token NLLs from a fresh non-overlap stride=2048 forward over val (mirrors 047B's `eval_val` methodology, including BOS-aware varlen attention via `cu_seqlens`), then calls the C scorer. Logs `ppm_full_native ... mix_bpb_sidecar=X.XXXXXX` and `ppm_native_submission_val_bpb: X.XXXXXX`.

4. **Hyperparameters** — adds 9 PPM env-var-driven fields to `class Hyperparameters`.

5. **Hook in `train_and_eval`** — after the diagnostic quantized eval, if `h.ppm_native_enabled`, calls `run_ppm_native_pass(h, device, val_data, eval_model)`.

6. **Imports** — `import ctypes, tempfile` added to top.

Total addition: ~400 lines including the C source string. The code is mechanically derivable from `/tmp/ppm_diff/patch_047B_ppm.py`.

**Critical compile gotcha:** `torch._dynamo.reset() + torch.cuda.empty_cache()` BEFORE the PPM forward's `torch.compile`. Otherwise the prior eval_val compile cache collides and `FailOnRecompileLimitHit` aborts the run. Validated in Phase 1.

## Hardware ladder

- **Mini rung: SKIPPED.** No training dynamics to validate; eval-only change. Phase 1 already validated end-to-end on 1×H100 EVAL_ONLY → got 1.00506 sidecar.
- **Official rung: 8×H100 SXM, EVAL_ONLY mode**, 600s budget. PPM scoring is CPU-bound (~17s OMP-8) and runs on rank 0 only; forward parallelizes across ranks. Predicted total eval time ~150–200s.

## Seed plan

3 seeds: **42, 7, 1337** (matches 1850 exactly for head-to-head). Each loads the SAME `final_model.int6.ptz` from spec 047B; the seed only affects PPM eval determinism (which is roughly seed-independent — PPM is byte-deterministic given the input stream).

In practice: a single seed should give the same number to within 0.0001 BPB. Running 3 mainly proves no submission-pipeline flakiness.

## Inputs

- **Model checkpoint:** `/workspace/runs/047B-loop-kv-shrink-screen/final_model.{pt,int6.ptz}` — existing, NO retraining. The `.pt` is needed as SpinQuant template; the `.int6.ptz` is the actual quantized weights.
- **Val data:** `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin` and matching `fineweb_val_bytes_*.bin` sidecar.
- **Tokenizer:** `…/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`.
- **Train log to seed env vars from:** `runs/047B-loop-kv-shrink-screen/train.log`.

## Checkpoints emitted

None new. Reuses 047B's existing `final_model.int6.ptz`. Per-run artifacts:
- `runs/052-ppm-port-tuned/seed_{42,7,1337}/eval_ppm.log` — full stdout including `ppm_native_submission_val_bpb:` line
- `runs/052-ppm-port-tuned/seed_{42,7,1337}/submission.json` — produced by the existing serialize flow if EVAL_ONLY can be made to emit it; otherwise constructed manually post-hoc

## Stop-early criteria

- If first seed `mix_bpb_sidecar > 1.015`, abort the other two seeds (something's drifted from Phase 1's 1.00506).
- If `eval_time > 550s`, abort and switch to OMP chunked (`PPM_OMP_CHUNK_TOKENS=4194304`, expected ~0.005 BPB worse but ~5× faster scoring).
- If C compile fails, abort — the wrapper script raises explicitly per 1850's "no silent fallback" policy.

## Cost estimate

- 1×H100 verification rerun on `seed_42` (sanity): ~$0.30, ~5 min.
- 3-seed 8×H100 official submission: ~$3–4 total, ~5 min wallclock per seed.

## Extra artifacts (beyond EXECUTION.md defaults)

- `eval_ppm.log` per seed (full stdout including PPM stats)
- The patched `train_gpt.py` itself, frozen at the spec's commit hash, lives at `records/track_10min_16mb/2026-04-28_047B_PPM_Tuned/train_gpt.py`

## Open questions for interview

1. **Where to place the patched `train_gpt.py`?** Options: (a) new record dir `records/track_10min_16mb/2026-04-28_047B_PPM_Tuned/`, (b) in `runs/052-ppm-port-tuned/` only. Convention says (a) so future submissions can `git checkout` it.
2. **Single-pass vs OMP-chunked at submission?** Single-pass gains an additional ~0.005 BPB but takes ~150s (vs ~17s OMP). Both fit budget; recommend single-pass since the gain is real and we have headroom. Default: `PPM_OMP_CHUNK_TOKENS=0`.
3. **Submission format**: do we need to wrap the result in `submission.json` like 1850 does, or is the log line sufficient? Worth confirming with execution.
4. **Multi-rank distributed gather**: Phase 1 ran rank-0-only on 1×H100. For 8×H100 we either (a) run rank 0 only and skip 7/8 of forward work (slower but simpler), or (b) implement file-based gather as 1850 did. Recommend (b) for proper 8-GPU scaling.
5. **3 seeds or 1?** PPM is byte-deterministic given identical NN NLLs; multi-seed mostly tests submission-pipeline reliability. Could do 1 first to confirm, then 2 more if it succeeds.

## Phase 2 followups (NOT in this spec)

- Stack TTT on top: PR #1850 has no TTT; our analysis (`eval/2026-04-28_ppm-rigorous-full-val.md`) suggests 34% of PPM's gain mechanism overlaps with TTT (within-doc rare-term recall), but the other 66% (space-byte rescues + structural calibration) should compound. Spec 053?
- PPM order sweep (5, 6) — small extra gain hypothesized but not measured
- Try on a stronger NN base (PR #1797's 1.06157, or our 050 family once it stabilizes)
