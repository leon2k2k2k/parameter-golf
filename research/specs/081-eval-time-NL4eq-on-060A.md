# Spec 081 — Eval-time NL=4-equivalent recurrence on 060A checkpoint

**Status:** FROZEN — config-only on `exp/071-loop-pattern @ e7ccda2`,
eval-only on the saved 060A checkpoint. No new code beyond what 080 uses.

**Date:** 2026-04-29 (autonomous wake W2)
**Branch:** `exp/071-loop-pattern` (same as 080; LOOP_PATTERN env var)
**Pinned commit:** `e7ccda278b76b093990191c53408c615fb21c05e`
**Parent run:** 060A's saved `final_model.pt`. Sibling to spec 080.
Source: `research/ideas/parallelize-deep-looks.md` cluster H/M (W2).

## Hypothesis

Companion to spec 080 (NL=3 equivalent at eval). 080 tests "one extra
pass at eval — does it help?" 081 tests **"how many extra passes
before the contraction breaks?"** by going one step further: 5 passes
through {3,4,5} (= NL=4 equivalent), 23 total layer-passes.

Per Two-Scale Latent Dynamics (2509.23314), loop-step sizes shrink
rapidly; the model output should asymptote rather than diverge as we
add passes. So:

- If 080 helps slightly and 081 helps more → recurrence is genuinely
  contracting toward a better fixed point and we left bpb on the table
  by training at NL=2.
- If 080 helps and 081 plateaus or regresses slightly → we found the
  practical limit of free deeper-recurrence.
- If 080 hurts and 081 hurts more → trained NL is the right point;
  closes "free deeper recurrence" thesis.

Reading 081 alongside 080 builds a **scaling curve** rather than a
single point. Information value > $1 cost.

## Baseline

Same baseline as 080: 060A pre-quant post-EMA val_bpb at canonical
NL=2 (~1.06358, exact value in `runs/060A-1855-port/seed_42/final.json`
when sampled).

## Expected Δ vs 060A

Tighter than usual since weights are unchanged:

| Outcome | Δ bpb | Likelihood |
|---|---|---|
| Best (deeper recurrence is free, 081 ≥ 080's gain) | −0.0005 to −0.0030 | low |
| Plateau (080 helped, 081 ≈ 080) | −0.0001 to +0.0005 | medium |
| Regression (over-iteration overshoots) | +0.0005 to +0.0040 | medium |

## Accept criteria

- **Win:** pre-quant val_bpb ≤ **1.06330** (matches 080 win threshold)
- **Noise:** 1.06330 – 1.06430
- **Kill:** ≥ 1.06430

## Config diff vs 060A baseline

```
LOOP_PATTERN = "1,2,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,6,7"
NUM_LOOPS    = 2
LOOP_START   = 3          # ignored when LOOP_PATTERN is set
LOOP_END     = 5          # ignored when LOOP_PATTERN is set
ENABLE_LOOPING_AT = 0.0   # eval-only
RESUME_FROM_CKPT = /workspace/runs/060A-1855-port/seed_42/final_model.pt
```

Pattern body: 19 visits across layers 1-7. Combined with pre `[0]`
and post `[8,9,10]`: **23 total layer-passes.** Layers 3,4,5 each
visited 5 times (= NL=4 equivalent).

Index lists (verified):
- `encoder_indices = [0,1,2,3,4,5,3,4,5,3,4]` (11 indices)
- `decoder_indices = [5,3,4,5,3,4,5,6,7,8,9,10]` (12 indices)

## Code changes

**None.** Same `LOOP_PATTERN` env var introduced at `e7ccda2` for 071,
re-used by 080. Different pattern string, identical code path.

### Compile-graph audit

Eval-only run. Model construction reads `LOOP_PATTERN` once at
`GPT.__init__`, building the 23-element index lists baked-in at
module construction. Compiled `forward_logits` traces once on the
first eval batch (one new graph variant due to longer iteration
count vs training-time NL=2 graph). From the second eval batch
onward, the graph is cached. **No mid-run recompile.**

Per memory `feedback_loop_activation_compile_is_cache_not_recompile`:
this single first-batch compile is *cache production*, not mid-run
recompile — let it finish.

Bank weights load from 060A checkpoint via `load_state_dict`. Bank
shapes are layer-indexed, not pass-indexed: extra visits to {3,4,5}
re-read the same loaded rows. **No weight slicing in compile.**
**No narrow-K matmul.** **No dynamic shape changes.**

Audit passes per the file's hard rules.

## Hardware ladder

- **Single rung: 4×H100, eval-only.**
- Wall: ~10-15 min (compile burst longer than 080 due to 23-pass graph;
  expected one compile, then eval runs at the new step rate).

## Seed plan

1 seed (42) — matches 060A checkpoint.

## Inputs / Outputs

- Same as 080. Output dir: `/workspace/runs/081-eval-NL4eq/seed_42/`.

## Stop-early criteria

- Pre-quant val_bpb > 1.080 → kill (model broke under deeper iteration)
- Compile time > 8 min → kill (graph variant too expensive)
- NaN in logits → kill

## Cost estimate

~$1.50 (eval-only, ~12 min on 4×H100).

## Open questions

1. **Is `runs/060A-1855-port/seed_42/final_model.pt` on the pod?** Same
   dependency as 080.
2. **Run 080 first; 081 only meaningful if 080 has a measurable signal.**
   If 080 is exactly noise, 081 still informs the scaling shape but is
   lower priority.
3. **TTT/GPTQ off** for the cleanest pre-quant comparison (same as 080).

## Reading 080 + 081 together

Building a scaling curve at no incremental engineering cost:

| Spec | NL eq | Total layer-passes | Layers {3,4,5} visits each |
|---|---|---|---|
| 060A baseline (training NL) | 2 | 17 | 3 |
| **080** | **3** | **20** | **4** |
| **081** | **4** | **23** | **5** |
| 082 candidate (W3) | 1 | 11 | 1 |
| 083 candidate (W3+) | 5 | 26 | 6 |

The scaling curve tells us: optimal eval-NL, slope of bpb vs NL, and
whether the trained model is over- or under-iterated.

## Followup specs gated on result

- If 081 wins by more than 080: spec 083 (NL=5 eq), keep climbing.
- If 081 plateaus or regresses: optimal eval-NL ≈ 080's setting.
- Cross-reference with spec 082 (NL=1 eq) for the full curve.

## See also

- `research/specs/080-eval-time-deeper-loop-on-060A.md` — direct sibling
- `research/ideas/parallelize-deep-looks.md` cluster H/M
- `tmp_exec/launch_080_eval_deeper_loop.sh` — template for this launch
