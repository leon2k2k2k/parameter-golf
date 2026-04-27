# Spec 047D — Per-pass AdaLN on loop layers (replaces loop_iter_embeds)

**Slug:** `loop-adabn-screen`  
**Created:** 2026-04-27  
**Status:** READY (fix4 — re-warmup at loop activation, 2026-04-27)
**Branch:** `exp/047D-loop-adabn`  
**Commit:** `06634a5` (fix4: re-warmup + ENABLE_LOOPING_AT=0.35; prior: 7b38f8d = fix3)
**Links to:** `research/ideas/loop-ffn-expressivity.md`

## Hypothesis

`loop_iter_embeds` adds one additive offset to x at the *entry* of each pass — it
gives the attention and FFN a pass-index signal but only via the input residual.
A per-pass (γ, β) applied *to the residual contribution* of each branch is strictly
richer: γ modulates how much each branch contributes per pass (allowing pass 0 to
be coarse and pass 2 to do refinement), and β gives each branch a separate additive
shift per (pass, layer). Neither is redundant with the other — γ has no analogue
anywhere in the current model.

LoopFormer (arxiv 2602.11451) applies exactly this and gets +1.93pp zero-shot
accuracy vs a vanilla looped baseline at essentially zero parameter cost.

Running with `LOOP_ITER_EMBEDS=0` to isolate the effect cleanly.

## What changes

Four new parameters on GPT:
```
loop_adabn_γ_attn: [num_passes=3, num_looped=3, model_dim=512]  init 1
loop_adabn_β_attn: [num_passes=3, num_looped=3, model_dim=512]  init 0
loop_adabn_γ_mlp:  [num_passes=3, num_looped=3, model_dim=512]  init 1
loop_adabn_β_mlp:  [num_passes=3, num_looped=3, model_dim=512]  init 0
```
Total: 4 × 3 × 3 × 512 = **18,432 params** (~36KB FP16, ~9KB int4).  
Init is transparent — step 0 is byte-identical to AC-fix minus iter_embeds.

Applied in Block.forward for each loop layer call:
```python
# attn branch:
scaled_attn = attn_scale * attn_out
scaled_attn = γ_attn[p, li] * scaled_attn + β_attn[p, li]

# mlp branch:
scaled_mlp = mlp_scale * mlp_out
scaled_mlp = γ_mlp[p, li] * scaled_mlp + β_mlp[p, li]
```

## Fix history

- **eea2c67** (attempt 1+2): hung at backward pass of loop-active graph — NCCL/autograd deadlock.
- **ba41fc8** (fix1: dynamo.disable): same hang — disable on a staticmethod crashes inside compiled model.forward.
- **40a59db** (fix2: allow_in_graph module-level fn): completed cleanly but val_bpb=1.06520 ≈ baseline. Root cause: allow_in_graph marks the function opaque, blocking autograd from computing gradients to γ/β. They stayed at init forever.
- **ac6598b / 7b38f8d** (fix3: remove allow_in_graph + ENABLE_LOOPING_AT=0.0): plain elementwise ops are natively traceable. Also loop-active from step 1 to avoid mid-training hang.
- **f1370e7 / 06634a5** (fix4: re-warmup at loop activation + revert to ENABLE_LOOPING_AT=0.35): ported same fix as 047C `f474ad5` — `_run_cu_bucket_warmup()` called when looping_active flips to True prevents evicted-graph hang without needing 0.0. Now at 0.35 for clean baseline parity.

## Baseline

- Spec 045 armD (AC-fix): pre-quant **1.06479** with `LOOP_ITER_EMBEDS=1`
- This spec runs with `LOOP_ITER_EMBEDS=0` — fair comparison isolates AdaLN vs iter_embeds

## Expected Δ

- Pre-quant: −0.001 to −0.003 (LoopFormer analogy at similar scale; uncertainty high)
- If iter_embeds was load-bearing: could see +0.001 degradation from its removal,
  partially offset by AdaLN. Net unknown.

## Accept / kill criteria

- **Accept:** pre-quant bpb ≤ 1.06479 (matches or beats AC-fix)
- **Iterate:** 1.06479–1.06679 — AdaLN signal there but iter_embeds removal cost;
  try 047D+iter_embeds together
- **Kill:** pre-quant bpb > 1.06679

## Config diff vs AC-fix (045 armD)

```
LOOP_ADABN=1          # new
LOOP_ITER_EMBEDS=0    # was 1 — removed to isolate AdaLN effect
```

## Hardware ladder

- **4×H100, 20min screen** — TRAINING_ONLY_SCREEN=1, TTT_ENABLED=0
- New inductor compile needed (new per-pass conditioning in loop path)
- Prewarm seeded from e021255 cache (~12-15 min)

## Seed plan

Single seed (42) for screen.

## Inputs

- Train/val: standard fineweb10B SP8192 CaseOps paths  
- Code: `exp/047D-loop-adabn` @ `eea2c67`

## Checkpoints to emit

- `final_model.pt` (pre-quant EMA)
- No quantized artifact (TRAINING_ONLY_SCREEN=1)

## Stop-early criteria

- NaN at any step
- val_bpb > 1.15 at step 500
- Tok/s < 3,800,000 at step 100

## Cost estimate

~$1.50 (45 min @ ~$2/hr for 4×H100)

## Open questions for interview

1. `LOOP_ITER_EMBEDS=0` means the baseline comparison has a slight disadvantage
   (AC-fix used iter_embeds=1). If 047D fails, a follow-up 047D2 with both
   `LOOP_ADABN=1 LOOP_ITER_EMBEDS=1` would isolate whether AdaLN is additive.
2. The 18,432 params are stored as FP32 post-training (via restore_fp32_params).
   At GPTQ time these pass through _unbank_state_dict unchanged (not quantized).
   ~72KB unquantized overhead in compressed artifact. Acceptable for a screen;
   handle properly before 8H.
