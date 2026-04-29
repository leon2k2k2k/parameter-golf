# Spec 080 — Eval-time deeper recurrence (NL=3 equivalent) on 060A checkpoint

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`, eval-only
on the saved 060A checkpoint. No new code.

**Date:** 2026-04-29 (autonomous wake W1)
**Branch:** `exp/071-loop-pattern` (re-uses LOOP_PATTERN env var)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's saved `final_model.pt`. Source idea:
`research/ideas/parallelize-deep-looks.md` cluster H (W1).

## Hypothesis

The 060A model is trained with the canonical {3,4,5}NL=2 loop band
(17 layer-passes per step). At eval we have ~100-180s of unused
wallclock budget (per memory `project_eval_time_quant_repair_idea`)
and ~50% VRAM headroom. **This spec spends that headroom on extra
recurrence depth at eval time only**, leveraging the bank-indexed
weight architecture: visits to layers 3,4,5 reuse the same trained
banks regardless of how many times they appear in the index list.

Concretely: override `LOOP_PATTERN` at eval-time to encode 4 passes
through {3,4,5} (= NL=3 equivalent) instead of the trained 3 passes.

The trained model has only seen its iterative-refinement function
applied 3 times on the loop band. Running it 4 times tests:
1. Is the loop function a *contraction* (extra applications stay
   close to a fixed point — extra pass = small refinement)?
2. Or does it *diverge* (extra application moves output OOD —
   bpb worsens)?

Per Two-Scale Latent Dynamics (arXiv 2509.23314): *"loop-step sizes
shrink rapidly... mean ‖Δ(k)‖₂ decays quickly with k, typically
within 5-10 steps"*. This predicts extra passes are small refinements,
not divergent. So the bpb impact should be **bounded and small** — but
unclear whether the extra refinement is helpful (smoother prediction)
or hurtful (over-refinement, overshoot).

This is an empirical question with low cost to answer.

## Baseline

060A pre-quant post-EMA val_bpb at canonical NL=2 (17 passes):
**~1.06358** (memory `project_baseline_1851_post_1872` for the broader
context). The exact number for the seed-42 4H run is in
`runs/060A-1855-port/seed_42/launch.out` or `final.json`.

This spec compares against that exact 060A seed-42 measurement, since
they share weights — the only delta is eval-time iteration count.

## Expected Δ vs 060A baseline (eval-only comparison)

| Outcome | Δ bpb | Likelihood |
|---|---|---|
| **Best (extra refinement helps)** | **−0.0005 to −0.0020** | low-medium |
| Likely (recurrence is contracting; small change either way) | −0.0003 to +0.0005 | high |
| Worst (out-of-distribution; refinement diverges) | +0.0010 to +0.0030 | low |

Tighter than usual prediction range because **weights are unchanged** —
no LR-schedule, no random-seed, no training-dynamics noise. The
comparison is genuinely seed-matched.

## Accept criteria

- **Win:** pre-quant val_bpb ≤ **1.06330** (Δ ≤ −0.00028, ~1× SOTA std)
- **Noise:** 1.06330 – 1.06430
- **Kill:** ≥ 1.06430 (extra passes net-negative; trained-NL is the right point)

## Config diff vs 060A baseline

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,3,4,5,3,4,5,6,7"
NUM_LOOPS    = 2          # any positive value enables looping_active
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.0   # eval-only run; immediately active on first forward
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
```

The pattern body has 16 visits across layers 1-7. Combined with pre `[0]`
and post `[8,9,10]`: **20 total layer-passes** (vs canonical 17 = NL=2).
Layer-3,4,5 each visited 4 times (NL=3 equivalent).

## Code changes

**None.** Reuses `LOOP_PATTERN` env var introduced at `e7ccda2` for
spec 071 et al. RESUME_FROM_CKPT path uses the existing
`launch_060_eval.sh` infrastructure.

### Compile-graph audit

Eval-only run. Model construction reads `LOOP_PATTERN` once at
`GPT.__init__`, building `encoder_indices = [0,1,2,3,4,5,3,4,5]` and
`decoder_indices = [3,4,5,3,4,5,6,7,8,9,10]` (both Python lists, baked
in at module construction time, not tensors).

The compiled `forward_logits` (eval forward) is invoked for the first
time at the first eval batch. At that point, torch.compile traces the
forward and produces ONE graph variant for the new pattern's
iteration count. From the second eval batch onward, the graph is
cached — **no mid-run recompile.**

The bank weights (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`)
are loaded from the 060A checkpoint via `load_state_dict`. Shapes are
`[2 × num_layers, ...]` for qo/kv and `[num_layers, ...]` for MLPs;
they index by *layer index* (3/4/5), not pass count. Loading is shape-
identical to 060A. **No weight slicing in the compile region.**
**No narrow-K matmul.** **No dynamic shape changes post-loop-activation.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only via RESUME_FROM_CKPT.**
- Wall: ~10 min total (compile burst + ~5 min eval).

## Seed plan

1 seed (42) — matches the 060A checkpoint seed.

## Inputs

- Train data: not used (eval-only, no training pass).
- Val files / val-bytes: standard 060A paths.
- Tokenizer: standard 060A path.
- Hotstart: 060A's `final_model.pt`.

## Checkpoints emitted

- `train.log` — full eval log including compile burst + val_bpb at NL=3 equivalent.
- (Optional) `final_model.int6.ptz` if `GPTQ_CALIBRATION_BATCHES > 0`.

Saved to: `/workspace/runs/080-eval-deeper-loop/seed_42_NL3eq/`.

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (model broke under deeper iteration)
- Compile time > 5 min → kill (suggests graph variant explosion; investigate)
- NaN in logits → kill

## Cost estimate

~$1 (eval-only, ~10 min on 4×H100).

## Open questions for interview

1. **Is `runs/060A-1855-port/seed_42/final_model.pt` on the pod?** If not,
   need to either rerun 060A or use a different saved checkpoint.
2. **GPTQ on or off for this run?** Default to OFF (`GPTQ_CALIBRATION_BATCHES=0`)
   — we want to measure pre-quant bpb at the new NL, not re-do
   quantization. Saves time.
3. **TTT on or off?** Default OFF (PHASED_TTT_ENABLED=0) for the same
   reason — isolate the deeper-recurrence effect from TTT effects.

## Followup specs gated on result

- If win (Δ ≤ -0.0003): write 080-promo running with TTT+GPTQ on the
  full submission pipeline, see if pre-quant gain transfers post-quant.
- If win + TTT compatible: try NL=4 equivalent (5 passes through
  {3,4,5}) — does the gain scale? Spec 081.
- If kill: closes the "free deeper recurrence at eval" thesis. Trained
  recurrence count is the right inference point. Pivot to other clusters.
- Either way: H2 (blended NL=2 + NL=3 logits) becomes interesting if
  H1 lands close-to-baseline-noise.

## See also

- `research/ideas/parallelize-deep-looks.md` — parent ideas thread (cluster H)
- `research/specs/060A-1855-port-baseline.md` — baseline spec
- `research/specs/071-loop-pattern-tent-235.md` et al. — sibling LOOP_PATTERN specs
- `tmp_exec/launch_060_eval.sh` — eval-only launch template
