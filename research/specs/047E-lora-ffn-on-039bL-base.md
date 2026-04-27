# Spec 047E — Per-pass FFN LoRA on 039bL base

**Slug:** `lora-ffn-on-039bL-base`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/047C-per-pass-lora-ffn` (same code as 047C — only config differs)
**Commit:** `92ce3b7` (re-warmup at loop activation + ENABLE_LOOPING_AT=0.35)
**Links to:** `research/ideas/per-pass-lora-ffn.md`, `research/specs/047C-per-pass-lora-ffn.md`

## Hypothesis

047C tested per-pass FFN LoRA on the 045 armAC-fix stack (which includes LOOP_ITER_EMBEDS,
LOOP_SCALE_INIT=recip, RECUR_ALPHA_ENABLED=1, PHASED_TTT_NUM_PHASES=3). This spec runs the
same LoRA lever on the simpler 039bL baseline code (no iter_embeds, no recur_alpha, floor-
then-linear LR, PHASED_TTT_NUM_PHASES=1). Two motivations:

1. **Additive test.** If per-pass LoRA helps on the simpler 039bL stack, it likely helps on the
   045 stack too (and is orthogonal to the 045 improvements).
2. **039 continuation.** The 039bL line achieved 1.06594 and was abandoned when 045 took over.
   Adding LoRA to 039bL may recover some of the 045-stack gap (1.06594 → ~1.064?) without the
   full 045 feature set. If it does, the 047C + 047D combination on 045 stack becomes even more
   compelling.

## Baseline

**039bL (floor-then-linear, no-carry, seed 42), 4×H100, 20min (~1200s)**:
- pre-quant post-EMA val_bpb = **1.06594**
- quantized val_bpb = **1.07527**
- Code commit: `5bbf12f` (branch: research, `exp/039b-loop-band-activation-screen` @ `8f10d16`)

## Expected Δ

- Pre-quant: **−0.0010 to −0.0030** vs 1.06594
- Confidence: low-medium. No prior data at this code stack + LoRA combo.
  If 047C (on 045 stack) shows clear positive signal, confidence here rises.

## Accept criteria

- **Promote:** pre-quant Δ ≤ −0.0010 (beats 1.06584). Write eval, plan 3-seed 8H.
- **Iterate:** Δ ∈ (−0.001, +0.001): signal too weak — check 047C result first.
- **Kill:** Δ ≥ +0.001 (worse by noise margin or more).

## Code changes

**No new code.** Uses `exp/047C-per-pass-lora-ffn` @ `92ce3b7` — same LoRA implementation.

**Config diff vs 039bL:**

```bash
LOOP_FFN_LORA_RANK=2             # new; default 0
ENABLE_LOOPING_AT="${ENABLE_LOOPING_AT:-0.35}"  # standard; re-warmup handles hang
```

**039bL config to inherit (the differences from 045 armAC-fix):**

```bash
LR_SCHEDULE_MODE=floor_then_linear
SECOND_HALF_FLOOR_LR=0.1
RECUR_ALPHA_ENABLED=0            # 039bL did NOT have recur_alpha
PHASED_TTT_NUM_PHASES=1          # 039bL used phased_ttt_num_phases=1
LOOP_ITER_EMBEDS=0               # 039bL: no iter embeds
# LOOP_SCALE_INIT not set (default)
# No SLOPE_WARMDOWN
```

**Param count:** same as 047C (+92,160 LoRA params). See 047C spec for details.

## Hardware ladder

**4×H100, 1 seed.** No 8×H100 in this spec.

- Template: `--template-id y5cejece4j`
- Region: NE-1 preferred, JP fallback.
- Speed check at step 100–500: ≥ 4.30M tok/s (loop active from step 1).

**Cache:** fc54262 cache may be partially reusable (loop kernels differ due to LoRA delta path).
Accept `TRITON_AUTOTUNE_NUM_RUNS=1` for screen.

## Seed plan

Single seed: **42** (matches 039bL).

## Inputs

- Data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...`
- Tokenizer: fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
- Code: `exp/047C-per-pass-lora-ffn` @ `92ce3b7`
- Hotstart: none (train from scratch like 039bL)

Paths identical across JP and NE-1 per `[JP and NE-1 volume layout]`.

## Checkpoints to emit

- `train.log` — required
- `final_model.pt` (post-EMA, FP) — for comparison
- No INT6 quant artifact (TRAINING_ONLY_SCREEN=1)

Retention: under `runs/047E-lora-ffn-on-039bL-base/seed_42/`.

## Stop-early criteria

- NaN in train_loss or val_loss → stop.
- val_bpb > 1.15 at step 500 → stop.
- step_time > 1.5× baseline at step 100 → stop (LoRA forward issue).

## Cost estimate

- 4×H100 @ ~$5.50/hr × 25 min ≈ **~$2.30**
- No prewarm (screen only).

## Monitoring

- 1-min polling. Compare 047E vs 039bL baseline at matched steps.
- Auto-stop on completion, report final pre-quant val_bpb.

## Open questions for interview

1. **LR schedule compatibility.** `LR_SCHEDULE_MODE=floor_then_linear` is a 039b feature. Confirm
   the `92ce3b7` code supports `SECOND_HALF_FLOOR_LR`. If not, fall back to cosine (standard).
2. **Hotstart checkpoint.** 039bL had no hotstart; start from scratch.
3. **Cache reuse.** fc54262 cache has loop-active non-LoRA kernels. LoRA activation path is new →
   expect short autotune at step 1. TRITON_AUTOTUNE_NUM_RUNS=1 bounds cost.
