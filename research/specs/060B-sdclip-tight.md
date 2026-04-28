# Spec 060B — 046B-tight SDClip on 060A baseline (eval-only via RESUME_FROM_CKPT)

**Date:** 2026-04-29
**Branch:** `research` (config-only; no code change)
**Parent:** 060A (`final_model.pt` from `runs/060A-1855-port/seed_42/`)

## Hypothesis

Per spec 046's measurements, tightening MLP/ATTN/EMBED clip sigmas reduces quantization noise and gains **−0.00086 BPB** (046B-tight on prior baseline). The original 046B-tight ran into the 16 MB artifact cap (+184 KB over). With #1855's lrzip+brotli compressor (≈ −280 KB savings), the same SDClip tightening should now fit comfortably within 16 MB.

## Baseline

060A (1-seed): val_bpb expected in [1.0595, 1.0625]. Use exact measured number once 060A completes.

## Expected Δ

**−0.00086 BPB** vs 060A (research-measured on 045-armD base; expected to transfer with similar magnitude on #1855 base since LQER + GPTQ stack is identical).

## Accept criteria

- post-quant + post-TTT val_bpb ≤ (060A val_bpb − 0.0003)
- artifact size ≤ 15,990,000 bytes (≥10 KB margin)
- no GPTQ failure or NaN

## Config diff vs 060A

```
MLP_CLIP_SIGMAS:    11.5 (1855 default; UNCHANGED — this is already 046B-tight!)
ATTN_CLIP_SIGMAS:   13.0 → 12.5
EMBED_CLIP_SIGMAS:  14.0 → 14.5  (matches 046B-tight; was reduced from 15.0 in 1855 already)
```

**Wait — important note.** #1855 already sets `MLP_CLIP_SIGMAS=11.5` and `EMBED_CLIP_SIGMAS=14.0` (≈ 046B-tight values). So 060B's incremental over 060A is just `ATTN_CLIP_SIGMAS=12.5` (was 13.0). The big SDClip win is already baked into 060A.

## Code changes

None. Pure env-var override + RESUME_FROM_CKPT path (already in our 046 lineage at commit `0ea6a97`).

## Hardware ladder

- 4×H100, eval-only mode (`RESUME_FROM_CKPT=1`)
- ~5-7 min wall, ~$1-2 cost

## Seed plan

1 seed: 42 (matches 060A).

## Inputs

- Hotstart: `/workspace/runs/060A-1855-port/seed_42/final_model.pt`
- Train data: same as 060A
- Tokenizer: same

## Stop-early criteria

- Artifact > 16,000,000 bytes → fail (compression assumption broken; investigate)
- post-quant val_bpb > 1.075 → kill (something broke)

## Cost estimate

~$2, ~10 min wall.

## Run command

```bash
SEED=42 RUN_LABEL=seed_42 \
  RESUME_FROM_CKPT=/workspace/runs/060A-1855-port/seed_42/final_model.pt \
  ATTN_CLIP_SIGMAS=12.5 \
  bash tmp_exec/launch_060B_run.sh
```
