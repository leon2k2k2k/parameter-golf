# Spec 055 — Full submission: 050 baseline + tuned PPM-D post-training (single file)

**Date:** 2026-04-28
**Branch:** `exp/055-050-with-ppm-fullrun` @ `c27be23`
**Idea:** `research/ideas/ppm-d-mixture-and-anti-hijack.md`
**Phase 1 reference:** spec 052 measured 047B + tuned PPM = 1.00506 sidecar (end-to-end). spec 053 (perpass-ffn-scale on 050 base) measured at 1.00278 with the same PPM. This spec is the clean 050 baseline + PPM, NOT stacking 053's perpass-ffn-scale.

## Hypothesis

Train spec 050's NN baseline (PR #1797 base + BOS-fix + BETA2=0.99 + SPARSE_ATTN_GATE_SCALE=0.5) from scratch on 8×H100, then apply our tuned PPM-D byte mixture (`thr=0.76`, `nn_skip_thr_nats=0.277`) at submission eval time. Single train_gpt.py: training and post-training live in the same file, separated only by the EVAL_ONLY/TTT_EVAL_ONLY branch.

Predicts a 3-seed mean `val_bpb ≈ 1.005`, on par with PR #1850's 1.00495. Beats it cleanly only if 050's NN beats 1850's by ~0.005+ BPB; we'd take parity.

## Baseline

Spec 050 baseline (`f5b8af8`): pre-quant EMA val_bpb ~1.067 / post-quant val_bpb ~1.077. PR #1850 submission: 1.00495 (3-seed mean), 1.00425 (best seed).

## Expected Δ

`mix_bpb_sidecar = 1.005 ± 0.003` (3-seed mean). The post-training delta is well-validated (~−0.075 BPB sidecar measured on 047B/052); the only source of seed variance is training noise on the NN side. Per 1850's logs, 3 seeds gave std 0.00072 — same order expected here.

## Accept criteria

- 3-seed mean `mix_bpb_sidecar ≤ 1.008` (within 1-σ of 1850)
- Best-seed `mix_bpb_sidecar ≤ 1.005` (matches 1850's mean)
- Total `eval_time` ≤ 600s on 8×H100. Phase-1 measured 207s (single-GPU); on 8×H100 expect ~150s with parallel forward.
- Issue #1017 conditions 1–4 satisfied (causal NN; PPM-D normalized; score-before-update; single L→R pass with no chunk-reset by default).
- All 3 artifacts under 16 MB.

## Config diff (vs 050 baseline defaults)

```
# Existing 050 hyperparameters unchanged (BETA2=0.99, SPARSE_ATTN_GATE_SCALE=0.5, etc.)

# New post-training PPM block (eval-time only; training path is identical to 050)
PPM_NATIVE_ENABLED=1            # NEW: enables the PPM hook in eval
PPM_ORDER=4                     # match 1850
PPM_LAMBDA_HI=0.9               # match 1850
PPM_LAMBDA_LO=0.05              # match 1850
PPM_CONF_THRESHOLD=0.76         # tuned (vs 1850's 0.9)
PPM_NN_SKIP_THR_NATS=0.277      # NEW knob: anti-hijack guard (= 0.40 bits)
PPM_LOG_CACHE_SIZE=1048576
PPM_OMP_THREADS=8
PPM_OMP_CHUNK_TOKENS=4194304    # OMP-chunked default; set 0 for single-pass (slightly better, ~17s vs ~17s)
SEED ∈ {42, 7, 1337}            # match 1850's seeds for direct comparability
```

All other env vars are seeded by the launch script from spec-050's standard env, then overridden as above.

## Code changes

Branch `exp/055-050-with-ppm-fullrun` adds, on top of 050's `records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py`:

1. **PPM C source** (~131 lines, embedded as `_NATIVE_PPM_C_SRC` r-string) — port of 1850's `score_byte` / `ppm_score` / `ppm_score_omp` plus our anti-hijack guard. The 5-line patch in `score_byte`:
   ```c
   int hi_raw = (conf >= thr);
   int hi = hi_raw && !(nn_skip_thr > 0.0 && nn_logp > -nn_skip_thr);
   double lam = hi ? lambda_lo : lambda_hi;
   (*gate_total)++;
   if (hi) (*gate_high)++;
   ```
   And `nn_skip_thr` parameter plumbed through `ppm_score` and `ppm_score_omp`.

2. **Python helpers** — `_build_native_ppm_lib()`, `_build_token_bytes_lut_for_ppm()`, `_ppm_mixture_bpb_native()`. Faithful port of 1850's wrappers, plus `nn_skip_thr` argument.

3. **`run_ppm_native_pass()`** — collects per-token NLLs from a fresh non-overlap stride=2048 forward over val (mirrors 050's `eval_val` methodology, including BOS-aware varlen attention via `cu_seqlens`), then calls the C scorer. Logs `ppm_full_native ... mix_bpb_sidecar=X.XXXXXX` and `ppm_native_submission_val_bpb: X.XXXXXX`.

4. **Hyperparameters fields** — 9 PPM env-var-driven fields added to `class Hyperparameters` after `eval_stride`.

5. **Hook in `train_and_eval`** — UNCONDITIONAL (placed after `deserialize`+`looping_active=True` setup, before the `if not ttt_eval_only:` block). This way it runs whether the model was just trained OR loaded from disk via TTT_EVAL_ONLY:

   ```python
   eval_model = deserialize(h, device)
   if h.num_loops > 0:
       eval_model.looping_active = True
   if h.ppm_native_enabled:
       log("\nbeginning PPM native pass")
       run_ppm_native_pass(h, device, val_data, eval_model)
   if not ttt_eval_only:
       compiled_model = torch.compile(eval_model, ...)
       timed_eval("diagnostic quantized", eval_val, ...)
       del eval_model
   ```

6. **Imports** — `import ctypes, tempfile, hashlib` added to top.

Total addition: ~400 lines including the C source string. Patcher script `tmp_exec/patch_055_ppm.py` (committed alongside the spec) generates the patched file from the canonical 050 baseline.

**Critical compile gotcha:** `torch._dynamo.reset() + torch.cuda.empty_cache()` is called BEFORE the PPM forward's `torch.compile(dynamic=True, fullgraph=False)`. This avoids `FailOnRecompileLimitHit` if a prior compile (e.g., training's compiled forward) is still cached. Validated in spec 052 / 053 phase-1 runs.

## Hardware ladder

- **Mini rung: SKIPPED.** Eval-time-only addition to a known-good NN base; no training dynamics to validate. Phase 1 already validated end-to-end on 1×H100 + an existing 047B-quantized blob (got 1.00506) and on 052-quantized (got 1.00278). 050 should land near 1.005.
- **Official rung: 8×H100 SXM**, 3 seeds, full training (600s) + eval. PPM scoring runs on rank 0 only; forward parallelizes across ranks (~150s expected for the PPM collection pass on 8×H100).

## Seed plan

3 seeds: **42, 7, 1337** (matches 1850 for direct comparability). Each seed trains a fresh 050 NN (training noise dominates seed variance), then runs PPM scoring on its own quantized model. PPM is byte-deterministic given identical NLLs.

## Inputs

- **Val data**: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin` and matching `fineweb_val_bytes_*.bin` sidecar (uint16).
- **Train data**: same path, `fineweb_train_*.bin`.
- **Tokenizer**: `…/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`.
- **Hotstart**: none (fresh training).

## Checkpoints emitted

Per seed:
- `final_model.pt` — pre-quant post-EMA (~135 MB)
- `final_model.int6.ptz` — post-quant submission blob (≤16 MB cap)
- `train.log` — full training + eval log including the `ppm_full_native` line and `ppm_native_submission_val_bpb:` final number

Optional (cached for downstream sweeps):
- `val_nlls_cached.npz` — per-token NLLs from the PPM forward pass, ~1.2 GB. NOT shipped in submission; saved to `runs/055-050-with-ppm-fullrun/seed_<N>/` for hyperparameter re-sweeps without re-running the model.

## Stop-early criteria

- If first seed's pre-quant EMA val_bpb > 1.075 by step 5000, abort the seed (training is broken; not a PPM problem).
- If first seed's `ppm_native_submission_val_bpb > 1.020`, abort the other two seeds (something drifted from spec-052 phase-1's 1.00506).
- If `eval_time > 550s`, abort and switch to OMP-chunked PPM (`PPM_OMP_CHUNK_TOKENS=2097152`, half the chunk size); single-pass is already chunked at 4M tokens.
- If C compile fails, abort — explicit raise per "no silent fallback" policy from 1850.

## Cost estimate

- 3-seed full submission on 8×H100: **~$15** (~5 min training + ~3 min eval per seed at ~$24/hr; 3 × 8 min ≈ 24 min total).
- Optional 1×H100 eval-only validation rerun (sanity check before submitting): ~$0.30, 5 min.

## Extra artifacts (beyond EXECUTION.md defaults)

- `train.log` per seed contains the full `ppm_full_native` line including:
  - `mix_bpb_piece`, `mix_bpb_sidecar` (the leaderboard-relevant number)
  - `nn_byte_bpb_piece`, `nn_byte_bpb_sidecar`
  - `ppm_only`, `gate_high_frac`
  - All hyperparameters echoed
- `submission.json` per seed (constructed by 050's existing serialize flow; no PPM-specific fields).

## Open questions for interview

1. **Single-pass vs OMP-chunked at submission?** Phase 1 used OMP-chunked (`PPM_OMP_CHUNK_TOKENS=4194304`, ~17s scoring on 1×H100). Single-pass (`=0`) is ~150s on 1×H100, gains ~0.005 BPB additional but tighter on cluster eval budget. Recommend chunked default; switch to single-pass if first seed has eval_time slack.
2. **Multi-rank gather for PPM**: Phase 1 ran rank-0-only on 1×H100. For 8×H100 we either (a) run rank 0 only and skip 7/8 of forward work (slower but simpler — OK if fits budget), or (b) implement file-based gather as 1850 did. Recommend (b) for proper 8-GPU scaling. Code path is in 1850's `run_ppm_native_pass` (line 3844 of their packed train_gpt.py); port it verbatim.
3. **Cache `val_nlls_cached.npz`?** Adds ~1.2 GB to artifact dir. Not shipped, but useful for post-hoc hyperparameter sweeps without retraining. Default: yes; flip to no if disk pressure.
4. **Training noise vs PPM determinism**: PPM is fully deterministic given identical per-token NLLs. So 3-seed std reflects only NN training noise. Std should be ~0.0007 (per 1850).
5. **Drift between 1×H100 forward and 8×H100 forward**: Phase 1 measured 0.00006 between our 1×H100 inspect_with_ppm forward and the official 8×H100 eval_val on 047B. Should be similar here.

## Phase 2 followups (NOT in this spec)

- **TTT stacking** — apply TTT after the diagnostic quantized eval, before PPM. Predicted: 50% of TTT's gain transfers (the part PPM doesn't already cover). Spec 056?
- **Stronger NN base** — port to 053 (perpass-ffn-scale) once that spec is evaluated. Already measured at 1.00278 in phase-1; promote to full submission spec 057?
- **PPM order sweep** — 5, 6 might give marginal gain on rare-term recall. Spec 058?
