# Spec 047 — KV group-of-4 screen (NUM_KV_HEADS=2)

**Slug:** `kv-group4-screen`  
**Created:** 2026-04-27  
**Status:** READY  
**Branch:** `exp/045-loop-layer-improvements`  
**Commit:** `e021255`  
**Links to:** `research/ideas/loop-layer5-attn-reduction.md`

## Hypothesis

Reducing NUM_KV_HEADS from 4 → 2 (GQA group size 2 → 4) shrinks the kv_bank
from [22, 256, 512] to [22, 128, 512], removing ~1.08M params. At 6-bit
quantization + brotli, this saves an estimated **600–700KB** in the compressed
artifact — enough headroom to use 046G-tightest DCLIP settings
(ATTN=11.5, MLP=10.5, EMB=13.5), which cost +689KB but give **−0.004 post-quant
bpb** vs baseline. Training quality impact is unknown and is what this screen
measures.

Config change is pure hyperparameter — no code modification needed.

## Baseline

- Spec 045 armD (AC-fix): pre-quant **1.06479**, post-quant **1.07387**
- Compressed size: **15,957,947 bytes** (42KB under 16MB cap)
- DCLIP: ATTN=13.0, MLP=12.0, EMB=15.0

## Expected Δ

- Pre-quant: −0.000 to +0.003 (GQA degradation unknown at this ratio)
- Post-quant (if tightest DCLIP applied): −0.003 to −0.001 net vs baseline
- Compressed size: ~15.3–15.4MB (estimated)

## Accept / kill criteria

- **Accept (proceed to 8H):** pre-quant bpb ≤ 1.06679 (+0.002 vs 1.06479)
- **Iterate:** pre-quant bpb 1.06679–1.06979 — small degradation, consider
  combining with MLP widening to compensate
- **Kill:** pre-quant bpb > 1.06979 (+0.005) — KV reduction too destructive

## Config diff vs AC-fix (045 armD)

```
NUM_KV_HEADS=2   # was 4 — only change
```

Everything else identical to AC-fix:
```
NUM_HEADS=8 MODEL_DIM=512 NUM_LAYERS=11
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35
LOOP_ITER_EMBEDS=1 LOOP_SCALE_INIT=recip
MLP_OUTER_ACTIVATION=leaky_relu_square NEGATIVE_SLOPE=0.5 SLOPE_WARMDOWN=-1.0
MATRIX_BITS=6 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 EMBED_CLIP_SIGMAS=15.0
```

## Code changes

None. `num_kv_heads` is fully parameterized:
- `kv_dim = h.num_kv_heads * head_dim` → 128 instead of 256
- `kv_bank: [22, 256, 512] → [22, 128, 512]`
- GQA attention already handles arbitrary `num_kv_heads`

## Hardware ladder

- **4×H100, 20min screen** — training-only, TRAINING_ONLY_SCREEN=1, TTT_ENABLED=0
- New inductor compile required (different kv_bank shape → new kernels)
- Inline prewarm: cold compile, ~25 min. Seed from any existing cache won't work.
- Effectively: 25 min prewarm + 20 min training = 45 min total

## Seed plan

Single seed (42) for screen. If accept, run 3 seeds at 8×H100.

## Inputs

- Train/val data: standard fineweb10B SP8192 CaseOps paths
- Checkpoint: none (train from scratch)
- Code: `exp/045-loop-layer-improvements` @ `e021255`

## Checkpoints to emit

- `final_model.pt` (pre-quant EMA)
- `final_model.int6.ptz` (GPTQ quantized)
- Retain both. Stop before 8H if pre-quant fails accept criterion.

## Stop-early criteria

- NaN in loss at any step
- val_bpb > 1.15 at step 500 (catastrophic failure)
- Tok/s < 3,800,000 at step 100 after compile (throughput regression from smaller kv_bank is expected to be minor or zero)

## Cost estimate

- 4×H100 screen: ~$1.50 (45 min @ ~$2/hr for 4×H100)

## Post-screen: size verification

After training, run eval-only with tightest DCLIP:
```
ATTN_CLIP_SIGMAS=11.5 MLP_CLIP_SIGMAS=10.5 EMBED_CLIP_SIGMAS=13.5
```
and report compressed size. If ≤ 16,000,000 bytes, path is confirmed.

## Open questions for interview

1. Does kv_bank shape change invalidate any checkpoint-loading logic (state dict
   keys assumed)? Should be fine but verify.
2. Throughput impact: fewer KV params = slightly faster attention. Expect small
   throughput gain. Confirm at step 100.
3. If quality degrades +0.001–0.003, is the tightest DCLIP gain (−0.004
   post-quant) still net positive? Yes — post-quant is what the leaderboard
   measures.
