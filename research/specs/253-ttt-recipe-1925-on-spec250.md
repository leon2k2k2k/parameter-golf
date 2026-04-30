# Spec 253 — PR #1925 TTT-recipe port on spec 250 outputs (eval-only)

**Date:** 2026-05-01 (deadline window)
**Idea:** PR #1925 simon-marcus's TTT-recipe update — single-phase TTT with
expanded prefix and lower LoRA LR. Demonstrated −0.00076 paired-3-seed BPB
on #1855 base.
**Lineage:** Spec 250 outputs (V21 + LeakyReLU² slope 0.3, commit d102266)
+ three eval-only env-var overrides. **Eval-only — no retrain, no code change.**

## Hypothesis

Phased-TTT saturation on V21+#1953 base is well-documented (250B/C/D/E all
flat) — but each of those is a *sub-step* of the same 3-phase recipe. PR #1925's
recipe is fundamentally different: **single-phase global-SGD** with **40% more
prefix data** and **20% lower LoRA LR**. This changes the optimization basin
TTT lands in, not just sub-step within it. Author demonstrated −0.00076 paired
3-seed Δ vs PR #1855's matched seeds (every seed individually better).

If the basin-shift mechanism is real and not stack-specific, it should produce
some recovery on top of V21+#1953-style multi-phase TTT. Our base is stronger
than #1855 (V21 vs plain), so the achievable Δ is likely smaller than the
−0.00076 demonstrated — but the **mechanism is the only TTT-side knob that has
demonstrated escape from the saturated landscape we observed in 250B/C/D/E.**

## Baseline

**Spec 250 same-seed post-TTT BPB.** Per-seed paired Δ on identical
{42, 0, 1234} seeds — noise cancels.

Reference per-seed targets (from spec 250 results):
- s42 post-TTT: **1.05797**
- s0 post-TTT: **1.05969** (warm)
- s1234 post-TTT: **1.05959**
- 3-seed mean: **1.05908** (σ 0.00079)

Reference for the recipe's claimed Δ: PR #1925 paired 3-seed = **−0.00076 BPB**
vs #1855 base.

## Expected Δ vs spec 250 (same seed)

**−0.0003 to −0.0008**, low-to-medium confidence.

- For: PR #1925's mechanism is a different TTT basin, not a sub-step. None of
  250B/C/D/E touched this basin, so the saturation finding doesn't apply.
  Author's matched-seed paired Δ was decisive (every seed beat baseline).
- Against: V21 + #1953-style (no_qv mask, TTT_LR_MULT=0.75, prefix=2500,
  3-phase) is itself a more aggressive TTT recipe than #1855's defaults, so
  the marginal improvement from switching to single-phase + prefix=3500 +
  TTT_LORA_LR=8e-5 may be smaller. Specifically the 3-phase recipe does
  ~3× more global-SGD on 2500 docs each = effectively more total adaptation
  than 1-phase on 3500 docs, so #1925's recipe is a **weaker** TTT regime
  in total adaptation terms — it may help by reducing TTT-overfit to val
  prefix, or hurt by under-adapting.

Author estimate of P(3-seed mean Δ ≤ −0.0003): **~30%**.
P(3-seed mean Δ ≥ 0): **~30%**.
P(small detectable win, 0 to −0.0003): **~40%**.

## Accept criteria

- **Per-seed:** post-TTT BPB at #1925 recipe ≥ −0.0003 below same-seed
  spec 250 result (above noise threshold given σ=0.0008).
- **3-seed mean:**
  - mean Δ ≤ −0.0005 → **promote**: submit spec 253. Stack clip arm next.
  - 0 ≥ mean Δ > −0.0005 → **wash**: submit spec 250 unchanged or fall back
    to #1953-verbatim path.
  - mean Δ > 0 → **kill**.
- All seeds clear 600s eval cap. Author reports eval time ~370s on similar
  stack — should fit easily even with prefix=3500.
- Artifact bytes unchanged from spec 250 (eval-only, no GPTQ rerun).

## Config diff (vs spec 250's eval defaults)

Three env var overrides; everything else verbatim from spec 250:

```bash
PHASED_TTT_NUM_PHASES=1     # was 3
PHASED_TTT_PREFIX_DOCS=3500 # was 2500
TTT_LORA_LR=0.00008         # was 0.0001 (default)
TTT_EVAL_ONLY=1             # use existing final_model.pt
RESUME_FROM_CKPT=1          # load spec 250's per-seed checkpoint
```

All other env (CASEOPS, EVAL_SEQ_LEN=2560, TTT_NO_QV_MASK=1, TTT_V_LORA=0,
QK_GAIN_INIT=5.25, TTT_LOCAL_LR_MULT=0.75, TTT_LORA_RANK=80, AWQ-lite, lrzip,
etc.) verbatim from spec 250.

**Note on TTT_LOCAL_LR_MULT vs TTT_LORA_LR:** these are different knobs.
- `TTT_LOCAL_LR_MULT=0.75` is a multiplier on the *per-doc LoRA optimizer*.
- `TTT_LORA_LR=0.00008` (default 0.0001) is the *global-SGD-phase LoRA LR*.
PR #1925 changes the latter. We keep both.

## Code changes

**None.** Eval-only via existing env-var support in spec 250's pinned commit
`d102266`. No new branch, no new commit.

## Hardware ladder

- **Mini (2×H100):** SKIP — eval-only, no model code change.
- **Single seed s42 first (8×H100, eval-only):** ~10 min, ~$1-2 → smoke +
  signal. If Δ < +0.0003 vs spec 250 s42, run s0 + s1234.
- **Full 3-seed (8×H100, eval-only):** ~30 min, ~$3-5 total.

## Seed plan

Three seeds {42, 0, 1234} matched to spec 250 for paired Δ. Run s42 first.

- If s42 post-TTT > 1.05900 (clearly regressing), abort.
- If s42 post-TTT in [1.05750, 1.05850], run s0 + s1234 sequentially.
- If s42 post-TTT < 1.05750 (clean win), run s0 + s1234 in parallel if pod
  capacity allows.

## Inputs

- **Source checkpoints:** `runs/250-1953-leakyrelu03/seed_<S>/final_model.pt`
  (per-seed, already on volume).
- **Quantized model:** `runs/250-1953-leakyrelu03/seed_<S>/final_model.int6.ptz`
  (already serialized; eval re-loads this).
- **Data + tokenizer:** verbatim from spec 250.
- **Commit:** `d102266` (same as spec 250).

## Checkpoints to emit

Per-seed `runs/253-ttt-recipe-1925-on-spec250/seed_<S>/`:
- `train.log` (eval-only, no train trajectory)
- `submission.json` (post-TTT val_bpb)
- No new `final_model.pt` (eval-only on existing checkpoint)

Retain logs ≥7 days for evaluation cross-checks.

## Stop-early criteria

- Eval BPB at TTT phase end > 1.06200 → abort (gross regression).
- Wallclock during TTT > 580s by phase 1 boundary → abort, cap-safety risk.
- Compile burst > 60s after eval start → flag, wait 1 min, abort if continues.

## Cost estimate

- Per seed: ~10 min eval-only, ~$1-2 on 8×H100.
- 3 seeds: ~$3-5 total.

## Extra artifacts

- Per-seed eval-time log at TTT chunk boundaries (existing `tttg:` log lines).
- Comparison table: spec 250 vs spec 253 paired per-seed in evaluation file.

## Open questions for interview

1. **Single-phase vs no-phased:** `PHASED_TTT_NUM_PHASES=1` means one global-SGD
   pass across the full 3500-doc prefix. Is the chunk schedule the same as
   spec 250's 3-phase × 2500-doc, just collapsed? Need to verify the eval
   loop's chunk-iter behavior matches author's claim.
2. **TTT_LORA_LR for global SGD vs per-doc:** verify in train_gpt.py that
   the env var maps to the global-SGD-phase LoRA LR (per #1925), not the
   per-doc LoRA LR (which is `TTT_LOCAL_LR_MULT × ttt_lora_lr_default`).
   If they're the same knob, the actual semantics may differ from #1925.
3. **Stack-with-clip:** if 253 wins, run spec 254 = 253 + MATRIX_CLIP=12.0
   eval-only. Pre-spec the next arm now or wait for results?
4. **Failure path:** if 253 lands wash, fall back to submitting spec 250's
   3-seed result (1.05908, regression vs #1953) or attempt to reproduce
   #1953 verbatim on our pod for an honest paired-comparison submission?
