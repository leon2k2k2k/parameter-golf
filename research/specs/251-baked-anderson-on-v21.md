# Spec 251 — Baked Anderson recurrence ported onto spec 250 V21+slope0.3 base

**Date:** 2026-05-01 (deadline window)
**Idea:** spec 112 baked Anderson (1.05968 single-seed on #1855 base) — port to V21 stack.
**Lineage:** Spec 250's V21+slope0.3 (PR #1953 verbatim + LeakyReLU² 0.5→0.3) + Anderson recurrence from spec 112 (e607dbe).

## Hypothesis

Spec 112 found that replacing canonical Markov-chain loop iteration with
Anderson mixing (frozen α from spec-111 LS-α inspection) hits **1.05968**
post-TTT single-seed on the #1855 base — beating PR #1935 SOTA (1.05997)
by 0.00029 at throughput parity. The architectural lever is small (≤120 line
diff, no new params) and orthogonal to V21's other levers
(AWQ-lite, AsymLogitRescale, 2560 ctx, no_qv TTT mask, TTT_LR=0.75).

If the architectural value is genuine and not specific to #1855's loop-band
activation distribution, baked Anderson on V21+slope0.3 should land below
spec 250's 1.0578 ± 0.0006 expected mean.

## Baseline

- **Spec 250** (PR #1953 + slope=0.3): expected 1.0578 ± 0.0006 (3-seed),
  pre-quant ~1.06187. Pinned commit `d102266` on `exp/250-1953-leakyrelu03`.
- **PR #1953** published: 1.05855 (3-seed mean).
- **Spec 112** reference (different base): 1.05968 (s42, #1855 base).

Direct paired comparison not possible — this spec is from-scratch retrain.
Reference is spec 250's same-seed result (when available) and the published
PR #1953 number.

## Expected Δ

**−0.0005 to −0.0020 vs spec 250 (same seed),** medium-low confidence.

- For: Anderson mixing was a genuine architectural improvement on #1855
  (spec 112). Mechanism is loop-band-only and orthogonal to V21's other
  levers (which touch quant, eval-time TTT, ctx length, MLP activation —
  none of these interact with the Anderson α coefficients).
- Against: α values (`[+0.57, +0.43]` for M=2; `[+0.55, -0.67, +1.12]`
  for M=3) were measured on **spec 111's #1855-based** trained-model
  loop activations. V21 has different MLP slope (0.3 vs 0.5), different
  embedding/lm_head paths (AsymLogitRescale at TTT), and different TTT
  regime (no_qv mask, LR=0.75) — all of which could shift the optimal
  mixing weights. The baked α is a static prior; real optimum on V21 may
  differ. **Risk: bake values mis-tuned for V21 → ≥0 Δ.**
- Composition risk: spec 116 showed that combining baked Anderson with
  frozen recur-α-β killed the win. V21's stack has its own learnable
  per-layer scaling (AsymLogit fine-tuned at TTT). Whether α-baked
  Anderson composes cleanly with all V21 levers simultaneously is
  untested.

Author estimate of P(win ≥ −0.0005 same-seed vs spec 250): **~25-35%**.

## Accept criteria

### Phase 1 — single-seed screen (s42, 8×H100)

- **Promote to Phase 2** if post-TTT BPB ≤ **1.05900** (≈ spec 250 mean
  1.0585 - 0.0005, conservative buffer for seed-42 lucky/unlucky variance).
- **Wash** if post-TTT BPB ∈ (1.05900, 1.06000): inconclusive at single
  seed; second seed needed to disambiguate. Promote conditionally if
  budget permits.
- **Kill** if post-TTT BPB > 1.06000: lever doesn't transfer to V21.
- Pre-quant sanity: must be within 1σ of spec 250's 1.06187 (i.e. ≤ 1.0625)
  — confirms training trajectory shape unchanged.
- All single-seed runs must clear 600s train cap, 600s eval cap, 16 MB
  artifact cap.

### Phase 2 — full 3 seeds (only if Phase 1 promotes)

- 3-seed mean post-TTT BPB **< 1.05825** (matching spec 250's accept
  criterion, ≥ 0.0003 below spec 250 expected mean).

If both phases land, this spec produces a stronger submission than spec 250
alone.

## Config diff (vs spec 250)

```
ANDERSON_ENABLED = 1               # was unset/0
ANDERSON_HISTORY = 2               # M=2 → 3 history entries
ANDERSON_BETA = 1.0                # unused with baked α
ANDERSON_REGULARIZATION = 1e-6     # unused with baked α
```

All other env vars verbatim from spec 250's launch script (CASEOPS_ENABLED=1,
EVAL_SEQ_LEN=2560, TTT_MASK=no_qv, TTT_LOCAL_LR_MULT=0.75, QK_GAIN_INIT=5.25,
ASYM_LOGIT_RESCALE=1, AWQ_LITE_ENABLED=1, etc.). See spec 250 for full env set.

## Code changes

- **Branch:** `exp/251-baked-anderson-on-v21` forked from
  `exp/250-1953-leakyrelu03` @ `274aa20`.
- **Pinned commit:** **`c8c2197`** (pushed to fork).
- **Train script:** `records/track_10min_16mb/2026-04-30_spec250_LongCtx_NoQV_LeakyRelu03/train_gpt.py`.
- **Diff scope:** 111 lines added, 2 modified — verbatim port of spec 112's
  e607dbe diff. Patches the same anchor points present in V21 (Hyperparameters,
  pre-Block module-level `anderson_step`, `class GPT.__init__` loop-band
  setup, `forward_logits` Anderson insertion, `forward_ttt` Anderson
  insertion, `BatchedTTTLoRA.__init__` num_slots accounting).
- **Patch applied cleanly** (`git apply --check` exit 0); Python AST parse
  passes.

## Hardware ladder

### Phase 1 — single-seed screen

- **Mini (2×H100):** SKIP. The Anderson change touches the loop-band
  forward graph and the TTT LoRA slot allocation; mini hardware can't
  exercise the full V21 + 600s wallclock + Phased TTT regime that's the
  signal. Compile validity is implicit in the syntax check + the
  bit-identical port from spec 112's already-validated commit.

  However, **a 4×H100 ~1200s matched-FLOPs run** is acceptable as a
  cheaper screen if 8×H100 capacity is tight: pattern from spec 112
  (4×H100 matched-FLOPs at ~$7) was sufficient to declare 1.05968.
  Caveat: 4×H100 doesn't include AsymLogitRescale's TTT-time SGD effect
  the same way (because TTT compute scales with GPU count), so the post-
  TTT number on 4×H100 will not be directly leaderboard-comparable.

- **Official (8×H100):** 1 seed (42), 600s train + ~470s eval. ~$12.

### Phase 2 — full 3 seeds (conditional)

- **Official (8×H100):** seeds 42, 0, 1234. ~$36.

## Seed plan

- **Phase 1:** seed 42 only. Mirrors spec 250's seed-first sequence and
  spec 112's 1.05968 baseline (also seed 42).
- **Phase 2 (only if Phase 1 promotes):** seeds 0 and 1234, sequential
  or parallel given pod availability.

## Inputs (from spec 250 reproduction)

Identical to spec 250. Specifically:
- `DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...`
- Tokenizer: `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- **No hotstart** — 600s wallclock-cap from-scratch training.
- **Required new env:** `ANDERSON_ENABLED=1 ANDERSON_HISTORY=2`
- **System dep:** `lrzip` binary (apt-get install lrzip) for
  `COMPRESSOR=pergroup` — same as spec 250.

## Checkpoints to emit

Per-seed final artifact only (`final_model.{pt,int6.ptz}` + `submission.json`).
No intermediate checkpoints — 600s training is too short, and we're
producing a submission, not a starting point.

Destination: `runs/251-baked-anderson-on-v21/seed_{42,0,1234}/`.

## Stop-early criteria

- **Per-seed:**
  - NaN/Inf in train loss → kill seed.
  - Step time > 130 ms (V21 reference ~121 ms) sustained 50 steps → kill
    seed (wallclock breach risk; Anderson should be throughput-parity but
    confirm in practice on V21).
  - **Mid-run torch.compile recompile** → kill seed
    (per memory `feedback_no_mid_run_recompile` — Anderson changes the
    compile graph; pre-warm verification is mandatory before launch).
  - Pre-quant BPB at stop_step > 1.072 (spec 250's 1.06187 + ~3σ on 0.001)
    → kill seed.
  - TTT compile fail (Anderson interacts with `BatchedTTTLoRA` num_slots
    counting) → kill seed and surface to research immediately; the LoRA
    slot accounting may have an off-by-one on V21's specific slot
    layout (V21 has `TTT_Q_LORA=0`, `TTT_V_LORA=0` which subtracts
    Q/V slots from each layer).
- **Across seeds (Phase 2):** if seed 42 post-TTT > 1.06000 in Phase 1
  re-run, abort 0 and 1234.

## Cost estimate

- **Phase 1 single-seed:** ~$12 (8×H100 × 30 min × $24/hr).
- **Phase 2 (conditional 2 more seeds):** ~$24.
- **Total max:** **~$36.** Phase 1 alone is ~$12 — bounded screen.

## Extra artifacts

- Per-seed `submission.json` matching spec 250 schema.
- Per-seed `train.log` (full Triton autotune output, for compile-graph
  validation since Anderson is a structural change).
- Per-seed `notes.md` summarizing Anderson-specific observations
  (compile burst length, throughput parity check, TTT slot layout
  validation).

## Open questions for interview

1. **Anderson value re-measurement?** The baked α was measured on spec
   111's #1855-trained model. On V21+slope0.3, the optimal α may differ.
   Should we (a) accept the spec 112 baked values as a screen and accept
   any V21-specific mismatch as the lever's transfer cost, or (b) first
   run an LS-α version on V21 to remeasure α before baking, paying
   ~$24 for one seed of "spec 111-port + LS-α active", then bake
   from there? Default (a) — single Phase 1 screen first, fall back
   to (b) only if Phase 1 washes.
2. **Pod region & capacity.** NE-1 first per memory, JP fallback.
   8×H100 only — do not silently substitute (memory:
   `feedback_dont_substitute_expensive_hardware`).
3. **Pre-warm requirement.** New code commit (Anderson) means a fresh
   inductor cache. Per memory `feedback_inline_prewarm_mandatory` and
   `feedback_prewarm_wallclock_for_loop_changes`, **inline Phase-0
   prewarm with MAX_WALLCLOCK_SECONDS≥900 and ENABLE_LOOPING_AT=0.05
   is mandatory** — Anderson modifies the loop-band forward graph,
   exactly the case that breaks 5-min KV-shrink-template prewarms.
4. **Phase 1 hardware.** 4×H100 matched-FLOPs ($7, faster turnaround)
   or 8×H100 official ($12, leaderboard-comparable)? Default 8×H100;
   spec 250's accept criterion is comparable to V21's 600s+8H regime,
   not 4H matched-FLOPs.
5. **Failure halting.** If seed 42 NaNs or post-TTT > 1.06000, halt
   Phase 2 yes/no? Default: yes.
6. **Submission decision.** If Phase 2 lands ≤ spec 250's 3-seed mean,
   submit spec 251 in place of spec 250. Confirm before submitting.
7. **Spec 250 vs spec 251 budget.** Both want 8×H100 official slots.
   Sequence: 250 first (lower risk, expected wins), 251 in parallel
   on a second pod if capacity, else sequential after 250 completes.

## Notes

- This spec is the **single most architecturally promising lever** in
  yesterday's recurrence-cluster research (spec 112's 1.05968 was the
  cluster's standout). Porting to V21 is the natural followup that
  yesterday's session ran out of time to do.
- This spec is a **sister** to spec 250 — both fork from V21+slope0.3
  base; 251 adds Anderson, 250 doesn't. Decision rule: submit whichever
  3-seed mean is lower, or spec 250 if 251 fails.
- Do not stack 251 with spec 250A (rank=56 eval-only). They're independent
  hedges and stacking introduces eval-side variance that complicates
  attribution.
