# Spec 047B — Loop-layer KV head reduction (LOOP_LAYER_NUM_KV_HEADS=2)

**Slug:** `loop-kv-shrink-screen`  
**Created:** 2026-04-27  
**Status:** READY  
**Branch:** `exp/048-loop-attn-shrink`  
**Commit:** `7d49b72`

## Hypothesis

Reducing KV heads only on the 3 loop layers (3,4,5) from 4→2 while keeping
non-loop layers at 4 is less destructive than reducing all 11 layers (spec 047).
The loop layers share weights across 3 passes — depth-recurrence ALBERT finding
suggests attention tying is approximately free, so halving KV capacity there should
hurt less than in non-loop layers where each layer sees data once.

Savings vs global 047: smaller parameter reduction, but the loop layers run 3×
more often — their KV bank gets more gradient signal per unit of capacity.

## Code change

**New parameter:** `loop_kv_bank: [2*num_looped, loop_kv_dim, model_dim]`  
= `[6, 128, 512]` (loop_kv_dim = 2 heads × 64 = 128)

`kv_bank` is unchanged `[22, 256, 512]` — loop layer rows exist but are never
used (zero gradient, no optimizer waste at the Muon level since kv_bank rows for
loops have zero grad).

`LOOP_LAYER_NUM_KV_HEADS=2` activates this; `=0` (default) is byte-identical to
baseline.

**Net new params in model:** +6×128×512 − 0 = +393,216 params above baseline.  
Wait — this ADDS a bank without removing anything. The actual param saving only
materializes if we also remove the loop rows from kv_bank (not done here for
screen simplicity). Screen is for quality signal only.

**GPTQ note:** `TRAINING_ONLY_SCREEN=1` is set — serialize/deserialize are skipped.
GPTQ wiring for `loop_kv_bank` is a TODO before any 8H run.

## Diff summary

```
+ loop_layer_num_kv_heads = int(os.environ.get("LOOP_LAYER_NUM_KV_HEADS", 0))
+ self.loop_kv_bank = nn.Parameter(torch.empty(2*num_looped, loop_kv_dim, model_dim))
+ # num_kv_heads patched on blocks[3..5].attn after construction
+ # _bank_weights dispatches to loop_kv_bank for i in [loop_start, loop_end]
+ # loop_kv_bank added to Optimizers.matrix_params, restore_fp32_params
+ # grad hook: scale all loop_kv_bank rows by 1/(num_loops+1)
```

## Baseline

- Spec 045 armD (AC-fix): pre-quant **1.06479**, post-quant **1.07387**

## Expected Δ

- Pre-quant: −0.001 to +0.002 (loop layers have shared weights → degradation may
  be smaller than global reduction)
- Size impact at GPTQ time: needs GPTQ wiring first; estimated −350KB compressed
  (loop_kv_bank rows at 128×512 instead of 256×512, ×3 loop layers ×2 K/V)

## Accept / kill criteria

- **Accept:** pre-quant bpb ≤ 1.06679 (+0.002 vs 1.06479)
- **Iterate:** 1.06679–1.06979 — degradation, compare vs 047 global result
- **Kill:** pre-quant bpb > 1.06979

## Config diff vs AC-fix (045 armD)

```
LOOP_LAYER_NUM_KV_HEADS=2   # new — only change
```

Everything else identical to AC-fix.

## Hardware ladder

- **4×H100, 20min screen** — TRAINING_ONLY_SCREEN=1, TTT_ENABLED=0
- New inductor compile required (loop_kv_bank is a new tensor shape in the graph)
- Inline prewarm seeded from e021255 cache (closest match)
- Expect ~15 min prewarm + 20 min training

## Seed plan

Single seed (42) for screen.

## Inputs

- Train/val: standard fineweb10B SP8192 CaseOps paths
- Checkpoint: none (train from scratch)
- Code: `exp/048-loop-attn-shrink` @ `7d49b72`

## Checkpoints to emit

- `final_model.pt` (pre-quant EMA) — for pre-quant bpb measurement
- No quantized artifact (TRAINING_ONLY_SCREEN=1)

## Stop-early criteria

- NaN at any step
- val_bpb > 1.15 at step 500
- Tok/s < 3,800,000 at step 100

## Cost estimate

- ~$1.50 (45 min @ ~$2/hr for 4×H100)

## Open questions for interview

1. The kv_bank loop rows (indices 3,4,5 and 14,15,16) exist but never receive
   gradient. Confirm Muon doesn't apply non-trivial updates to zero-grad rows.
2. GPTQ is not wired — confirm TRAINING_ONLY_SCREEN=1 so serialize is skipped.
3. If both 047 and 047B pass the accept gate, which to promote? The one with
   better pre-quant bpb. If comparable, 047 (global) saves more size at GPTQ.
