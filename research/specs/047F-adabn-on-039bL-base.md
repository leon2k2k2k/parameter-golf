# Spec 047F — Per-pass AdaLN on 039bL base (fix3 — no allow_in_graph)

**Slug:** `adabn-on-039bL-base`
**Created:** 2026-04-27
**Status:** READY
**Branch:** `exp/047D-loop-adabn` (same code as 047D fix3 — only config differs)
**Commit:** `7b38f8d` (fix3: remove allow_in_graph + SHA update)
**Links to:** `research/ideas/loop-ffn-expressivity.md`, `research/specs/047D-loop-adabn-screen.md`

## Hypothesis

047D fix3 tests per-pass AdaLN (γ, β conditioning on attn+MLP residual contributions) on the
045 armAC-fix stack. This spec runs the same lever on the simpler 039bL baseline to:

1. **Decouple from 045 features.** 047D fix3 on the 045 stack includes LOOP_ITER_EMBEDS=0,
   which removes iter_embeds to isolate AdaLN. On the 039bL base, iter_embeds never existed,
   so there's no confound — this is a clean additive test.
2. **Parallelism.** Running 047E and 047F simultaneously on 039bL costs ~$4.60 and gives two
   independent lever measurements on the same baseline. Strong correlation between the two
   results (both win or both lose) increases confidence that 045-stack experiments will agree.

## Baseline

**039bL (floor-then-linear, no-carry, seed 42), 4×H100, 20min:**
- pre-quant post-EMA val_bpb = **1.06594**
- quantized val_bpb = **1.07527**
- Code commit: `5bbf12f` (branch: research)

## Expected Δ

- Pre-quant: **−0.001 to −0.003** vs 1.06594
- Confidence: low. AdaLN at init is identity; signal requires gradient flow through γ/β.
  fix3 (removing allow_in_graph) should restore that flow. No prior clean measurement exists.

## Accept criteria

- **Accept:** pre-quant Δ ≤ −0.001 (beats 1.06484). Write eval, promote to 047D fix3 on 045 stack.
- **Iterate:** Δ ∈ (−0.001, +0.001): weak signal. Check if 047D fix3 on 045 stack wins.
- **Kill:** Δ ≥ +0.001. AdaLN is neutral even with correct gradient flow.

## Code changes

**No new code.** Uses `exp/047D-loop-adabn` @ `7b38f8d` — fix3 code (no allow_in_graph).

**Config diff vs 039bL:**

```bash
LOOP_ADABN=1                     # new
ENABLE_LOOPING_AT=0.0            # changed from 0.35 (prevents mid-run hang)
```

**039bL config to inherit:**

```bash
LR_SCHEDULE_MODE=floor_then_linear
SECOND_HALF_FLOOR_LR=0.1
RECUR_ALPHA_ENABLED=0
PHASED_TTT_NUM_PHASES=1
LOOP_ITER_EMBEDS=0               # 039bL never had iter_embeds — clean additive test
# LOOP_SCALE_INIT not set (default)
```

**New params:** 4 × [3 passes, 3 looped, 512] = 18,432 params. Init γ=1, β=0 → identity.
Gradient flow to γ/β is now enabled (fix3 removed allow_in_graph blocker).

## Hardware ladder

**4×H100, 1 seed.** No 8×H100 in this spec.

- Template: `--template-id y5cejece4j`
- Region: NE-1 preferred, JP fallback.
- Speed check: ≥ 4.30M tok/s at step 100 (loop active from step 1).

**Cache:** fix3 forward graph is identical to fix2 (`40a59db_adabn` stash). If stash is on
volume, restore for near-zero compile overhead. If not available, accept cold start (~15 min
autotune at step 1 — budget 900s for it or set `TRITON_AUTOTUNE_NUM_RUNS=1`).

## Seed plan

Single seed: **42** (matches 039bL).

## Inputs

- Data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...`
- Tokenizer: fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
- Code: `exp/047D-loop-adabn` @ `7b38f8d`
- Hotstart: none (train from scratch)

## Checkpoints to emit

- `train.log` — required
- `final_model.pt` (post-EMA, FP)
- No INT6 quant (TRAINING_ONLY_SCREEN=1)

Retention: under `runs/047F-adabn-on-039bL-base/seed_42/`.

## Stop-early criteria

- NaN → stop.
- val_bpb > 1.15 at step 500 → stop.
- Tok/s < 3.8M at step 100 → stop and investigate (AdaLN compute overhead).

## Cost estimate

- 4×H100 @ ~$5.50/hr × 25 min ≈ **~$2.30**
- May need ~15 min cold-start compile if fix3 cache not on volume → total ~40 min ~$3.65.

## Monitoring

- 1-min polling. Compare 047F vs 039bL baseline at matched steps.
- **Key diagnostic:** check γ/β gradient norm at step 1 (print loop_adabn_γ_attn.grad.norm()).
  If zero → fix3 didn't fix the gradient flow issue; stop immediately.
- Auto-stop on completion, report final pre-quant val_bpb.

## Open questions for interview

1. **LR schedule compatibility.** Verify `d9d6bcb` / `7b38f8d` code on `exp/047D-loop-adabn`
   supports `LR_SCHEDULE_MODE=floor_then_linear` and `SECOND_HALF_FLOOR_LR`. If not, use cosine.
2. **Fix3 cache.** If `/workspace/.inductor_cache_40a59db_adabn` is on volume and fix3's forward
   graph matches, restore it. Otherwise cold-start (budget 900s autotune).
3. **Gradient diagnostic.** Add one-line print to verify γ/β grad at step 1 before committing
   to a full 20-min run. This is the residual risk from fix3.
