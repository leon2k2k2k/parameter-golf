# Spec 060A — Port PR #1855 as new research baseline (1-seed validation)

**Date:** 2026-04-29
**Branch:** `research` (baseline-migration exception per `CLAUDE.md`)
**Parent context:** PR #1855 (codemath3000, val_bpb 1.06108 3-seed) was accepted by cocohearts in PR #1902 over current research baseline #1736 (1.06549). Our 050 stack inherits #1797 (1.06157) which is also under cloud per cocohearts. Spec 060 family adopts #1855's train_gpt.py as the new base for stacking quant-repair + deploy-time repair levers.

## Hypothesis

Porting PR #1855's `train_gpt.py` to our pod with seed 42 should reproduce a single-seed val_bpb in [1.0595, 1.0625], validating that our environment matches the cocohearts-accepted leaderboard chain. This spec is a pure port — zero code modifications — to establish the new baseline that 060B+ will stack on.

## Baseline

- PR #1855 (codemath3000): **1.06108** sidecar (3-seed mean, post-quant + post-TTT)
- Reported per-seed: not broken out in their README, but 3-seed std ~0.0007 per #1851/#1868 reproduction
- Our research baseline #1736: 1.06549
- Our spec 050: ~1.064 sidecar (never validated end-to-end with full TTT)

## Expected Δ

1-seed val_bpb in [1.0595, 1.0625]. Centered at 1.0610. Range allows for ±0.0015 pod/seed variance.

## Accept criteria

- Post-quant + post-TTT val_bpb ∈ [1.0595, 1.0625]
- `train_time ≤ 600s` (must hit wallclock cap, not iteration cap)
- `total_eval_time ≤ 600s`
- Artifact size **≤ 15,950,000 bytes** (gives ≥50 KB headroom for 060B's SDClip tightening; #1855 reports ~15.95 MB with their lrzip compressor)
- No NaN, no compile pathology, no GPTQ failure

## Config diff vs our 050 baseline

Effectively all 9 of #1855's hparam values come baked-in as defaults in their `train_gpt.py`:

```
BETA2 = 0.99                  (we have)
SPARSE_ATTN_GATE_SCALE = 0.5  (we have)
MLP_CLIP_SIGMAS = 11.5        (NEW — was 12.0 in 050)
EMBED_CLIP_SIGMAS = 14.0      (NEW — was 15.0 in 050)
WARMDOWN_FRAC = 0.85          (NEW — was 0.75 in 050)
TTT_BETA2 = 0.99              (NEW — was 0.999 default)
TTT_WEIGHT_DECAY = 0.5        (NEW — was 1.0 default)
TTT_LORA_RANK = 80            (NEW — was 96 default)
PHASED_TTT_PREFIX_DOCS = 2500 (NEW — was 2000 default)
```

Plus the **per-group lrzip+brotli compressor** (~280 KB savings vs plain brotli) — code lives in `_lrzip_compress` / `_lrzip_decompress` in #1855's train_gpt.py.

## Code changes

Pure port — zero modifications. Copy these files from PR #1855 (codemath3000/sp8192-lqer-bos-smear-fix-9hp-stack @ `1e43966`) into a new submission dir:

```
records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/
├── train_gpt.py          (3753 lines, verbatim from #1855)
├── lossless_caps.py      (verbatim)
├── prepare_caseops_data.py (verbatim)
├── README.md              (verbatim — rename author / acknowledge port)
├── submission.json        (rebuild for our seed)
├── tokenizers/
│   └── fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model (verbatim)
└── requirements.txt       (verbatim — note `lrzip` requirement)
```

**Branch:** `research` (baseline-migration exception). Commit hash: TBD on checkout.

**Why baseline-migration exception applies:** spec 060A's purpose IS to move the baseline. Subsequent specs (060B+) will fork `exp/060B-*` etc., but 060A lands on research as the new starting point.

## Hardware ladder

- **Mini rung: SKIPPED.** Pure port of an already-validated 3-seed submission. No code change to test.
- **Official rung: 8×H100 SXM** (matches #1855's reported config), 1 seed, 600s train + 600s eval budget.

## Seed plan

1 seed: **42** (matches one of #1855's three reported seeds for direct comparison if needed).

## Inputs

Standard paths (already on NE-1 / JP volumes):
- Train: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/.../fineweb_train_*.bin`
- Val: same dir, `fineweb_val_*.bin` + `fineweb_val_bytes_*.bin`
- Tokenizer: `…/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- Hotstart: **none** (fresh training)

## Checkpoints emitted

Per seed:
- `final_model.pt` — pre-quant post-EMA (~135 MB; not shipped)
- `final_model.int6.ptz` — post-quant submission blob (≤ 16 MB)
- `train.log` — full log including phased TTT eval line

Saved to: `/workspace/runs/060A-1855-port/seed_42/`.

## Stop-early criteria

- `train_loss > 5.0` at step 1000 → kill (training broken)
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- post-quant val_bpb > 1.085 (pre-TTT) → kill (TTT can't recover)
- Compile fail → kill
- NaN → kill
- `lrzip: command not found` → install on pod, restart

## Cost estimate

- 1-seed full run on 8×H100: **~$5** (~13 min wall = 600s train + ≈400s eval)

## Extra artifacts

- `train_seed42.log` — full per-seed log
- `submission.json` — single-seed metadata (rebuild for spec 060A identity)

## Open questions for interview

1. **Is `lrzip` installed on the parameter-golf pod template (`y5cejece4j`)?** #1855's compressor depends on it. If absent, the launch script must `apt install lrzip` early; otherwise serialize crashes.
2. **Pod region preference**: NE-1 first per memory, JP fallback. 8×H100 capacity drought has been an issue today — restart-loop may be needed.
3. **Stop pod after?** Yes per default policy. Keep warm only if 060B-E specs are ready to launch immediately after.
4. **Should we save intermediate checkpoints** (mid-training) for downstream specs that might want a hotstart? Not for 060A — 060B+ will retrain from scratch with their own arms.

## Phase 2 followups (NOT in this spec)

- **060B**: Add 046B-tight SDClip (MLP=11.5/ATTN=12.5/EMBED=14.5) — should fit within new lrzip headroom; expected −0.00086 BPB.
- **060C**: Add 046L deploy-time quant repair (commit `fcb816f`) — eval-side, free, expected positive but unmeasured.
- **060D**: Push to 046G-tighter SDClip (each clip −1.0σ) — needs additional ~150 KB compression headroom from 046I or LQER trim.
- **060E**: Full stack — 060B + 060C + 060D combined.
