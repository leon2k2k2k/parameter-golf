# Spec 060I — Port PR #1908 activation-aware mixed-bit GPTQ on 060A baseline

**Date:** 2026-04-29
**Branch:** `exp/060I-awq-lite` (forked from `research`)
**Parent:** 060A `final_model.pt` (training unchanged) + ~100 LOC port from PR #1908.
**Idea file:** `research/ideas/1908-awq-lite-mixed-bit-gptq.md`

## Hypothesis

PR #1908 (romeerp) reports **−0.000265 BPB 3-seed step-matched on PR #1855 itself** (1.06108 → 1.06081, std=0.00089) by adding activation-aware mixed-bit GPTQ: one 64-col weight group per layer protected at int8 (rest stays int6), selected by `act_rms * mean(|w|)` saliency.

Our 060A is a 1-seed reproduction of #1855 (val_bpb 1.07212 quantized, 1.05918 post-TTT). Porting this lever and re-quantizing 060A's `final_model.pt` should reproduce a similar quantization-tax recovery (~−0.0002 BPB) on our base.

## Baseline

060A.

## Expected Δ

- **Quant-tax recovery (post-quant, pre-TTT):** −0.0001 to −0.0003 BPB (author reports −0.000196 mean)
- **Post-TTT:** −0.0002 to −0.0004 BPB (author reports −0.000265 mean)
- Artifact size: +~16 KB (15,902 KB → ~15,918 KB; well under cap)

## Accept criteria

- Smoke (1 seed): post-TTT val_bpb improves over 060A by ≥ 0.0001
- Smoke: artifact ≤ 16,000,000 bytes
- Smoke: GPTQ doesn't crash or NaN; quantized_bpb sanity within 0.001 of pre-AWQ quantized number on same checkpoint
- 3-seed (Phase 2): mean delta vs 060A ≥ 2σ (where σ ≈ 0.00131 from #1855 reproduction)

## Config diff vs 060A

```
AWQ_LITE_ENABLED=1
AWQ_LITE_BITS=8
AWQ_LITE_GROUP_TOP_K=1
AWQ_LITE_GROUP_SIZE=64
```

(All other env vars match 060A / `tmp_exec/launch_060_eval.sh` defaults.)

## Code changes

Port from PR #1908's `train_gpt.py` into our `records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py`. Minimal, additive:

1. **`_awq_lite_group_candidates(w, act_rms, group_size)`** helper — line 2536 in PR #1908 diff. ~15 LOC. Pure helper, no dep on rest of the patch.
2. **Activation-stat collection** during GPTQ calibration — register a hook on each quantized linear's input, accumulate per-input-channel RMS into `act_stats[name]`. ~10 LOC.
3. **`awq_selected` set + per-group bit override** in the GPTQ quantize loop — line 2557-2630 in PR #1908 diff. Adds `protect_clip_range` and `group_size` kwargs to the existing per-tensor quantize call. ~30-40 LOC.
4. **Hyperparameter fields** on the config dataclass — 4 env vars wired through. ~10 LOC.
5. **(Optional)** `FORCE_STOP_STEP` for future matched-step eval. ~5 LOC. Skip in this spec; not needed since we use RESUME_FROM_CKPT.

Total: ~80-100 LOC, all in serialize/quantize path. No training-time code touched.

**Reference:** clone PR #1908's diff into a worktree and lift the `_awq_lite_*` functions verbatim where possible. Do NOT pull their full file — we want minimal diff vs our 060A baseline so any regression is traceable.

## Hardware ladder

- Smoke: 4×H100, eval-only via RESUME_FROM_CKPT (~5-10 min, ~$1-2)
- Phase 2: 8×H100, full retrain × 3 seeds for matched-step measurement (~$24, only if smoke passes accept criteria)

## Seed plan

- Smoke: seed 42, RESUME_FROM_CKPT=060A's final_model.pt
- Phase 2 (conditional): seeds 42, 0, 1234 — same as #1855's published suite, full retrain

## Inputs

- Parent ckpt: `/workspace/runs/060A-1855-port/seed_42/final_model.pt` (or `seed_42_4h/` if that's what we have)
- Train code: fork of `records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py` on `exp/060I-awq-lite`
- Tokenizer + dataset: same NA-1 paths as 060A

## Checkpoints to emit

- `final_model.int6.ptz` — re-quantized artifact (the actual submission file)
- `final_model.pt` — copy of 060A's pt (for chain-of-custody)
- `act_stats.pt` — saved activation RMS per layer (debug; <100 KB)
- `awq_selected.json` — which (layer, col_start, col_end) groups got int8 (debug; <10 KB)

## Stop-early criteria

- post-quant val_bpb > 1.080 → kill (catastrophic regression; AWQ port broken)
- GPTQ NaN / serialize failure → kill, debug locally
- artifact > 16,000,000 → fail-hard; reduce TOP_K or kill arm

## Cost estimate

- Smoke: ~$2
- Phase 2 (3-seed retrain): ~$24
- Total worst case: ~$26

## Open questions for interview

1. **Smoke target:** 060A `seed_42` (8H, leaderboard-valid) or `seed_42_4h` (4H, research-only)? Recommend 8H seed if available; 4H smoke is cheaper but throughput-mismatched.
2. **Phase 2 trigger:** auto-run if smoke ≥ accept criteria, or pause for review? Recommend pause (3-seed retrain is the expensive part).
3. **Stacking with 060B (ATTN 12.5):** spec as separate 060K *after* 060I lands, or piggyback into 060I Phase 2 with 060B's clip override? Recommend separate — cleaner attribution, only +$2 for an eval-only smoke of the stack.

## Phase 2 plan (NOT in this spec)

- If 060I passes 3-seed: spec **060K = 060I + 060B clip override** (single arm, 1-seed smoke + 3-seed if it lands).
- If both pass: that's our submission candidate.
