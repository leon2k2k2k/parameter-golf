# Spec 046 — Loop Depth & Width + C Screen

**Slug:** `046-loop-depth-width-c-screen`
**Created:** 2026-04-26
**Status:** READY
**Branch:** `exp/045-loop-layer-improvements`
**Commit:** `1c6cd7cea0fa18ab07de120285c989df7c86b6bb`
**Links to:** `research/ideas/loop-layer-improvements.md`, `research/ideas/loop-recurrence-normalization.md`

## Hypothesis

Lever C (1/L residual init) gave a clear systematic training loss advantage throughout
spec 045 (−0.003 to −0.010 at matched steps, biggest gain at loop activation). Two natural
extensions — going deeper (more passes, same window) and wider (more layers, same passes):

**Arm A — Loop45 NL=3 + C:** 041K showed NL=3 is the sweet spot over NL=2.
Running it with proper 1/L init (1/4 instead of 1.0) should compound: the residual
calibration problem is *worse* at NL=3 (4 passes vs 3), so C's benefit should be larger.

**Arm B — Loop345 NL=2 + C:** Expand loop window from layers 4–5 to layers 3–5.
Layer 3 has never been looped. Same total forward compute as Arm A (17 layer-passes each).
1/L init = 1/3 (3 passes through layers 3–5). Higher risk, novel territory.

## Baseline

`runs/039-neg-slope-screen-on-1797-base/baseline/`
Pre-quant EMA val_bpb: **1.06514** | Quantized: **1.07410** | Steps: 5156

## Expected Δ

| Arm | Expected Δ | Confidence |
|---|---|---|
| A (Loop45 NL=3 + C) | −0.001 to −0.003 | Medium — two validated positives stacked |
| B (Loop345 NL=2 + C) | −0.001 to −0.004 | Low — novel window, unknown risk |

## Accept criteria

- **Win:** pre-quant EMA ≤ 1.0641 (Δ ≤ −0.00104)
- **Noise:** 1.0641 – 1.0670
- **Kill:** ≥ 1.0670

## Config diff

All other config identical to baseline (039b). Both arms use `LOOP_SCALE_INIT=recip`.

```
# Arm A: deeper — same window (layers 4–5), one more pass, 1/L init = 1/4
NUM_LOOPS=3  LOOP_START=4  LOOP_END=5  LOOP_SCALE_INIT=recip

# Arm B: wider — expanded window (layers 3–5), same pass count, 1/L init = 1/3
NUM_LOOPS=2  LOOP_START=3  LOOP_END=5  LOOP_SCALE_INIT=recip
```

Both have identical total forward compute: 17 layer-passes post-loop vs baseline's 15.
Expected post-loop throughput: ~2.7M tok/s (vs baseline's ~3.3M).

## Code changes

No new code. Uses existing `exp/045-loop-layer-improvements` branch.
`LOOP_SCALE_INIT`, `NUM_LOOPS`, `LOOP_START`, `LOOP_END` are all env-var configurable
in the 045 commit. `LOOP_ITER_EMBEDS=0` and `MLP_ONLY_FROM_PASS=0` (defaults) for both arms.

## Hardware ladder

**Mini (this screen):** 4×H100, 20 min wallclock, seed=42, 2 parallel pods.
- Cost estimate: ~$7/arm × 2 arms ≈ **$14**.

**Official:** 8×H100, 3 seeds. Only if arm clears mini accept threshold.

## Seed plan

Mini: seed=42. Official: seeds 42, 314, 1337.

## Inputs

- Data: `/workspace/parameter-golf/data/`
- Tokenizer: `/workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/tokenizer.model`
- No hotstart. Config: 039b baseline + arm-specific env vars above.

## Checkpoints to emit

Pre-quant EMA + quantized blob. No optim state on mini.

## Stop-early criteria

- NaN at any val eval → stop
- val_bpb > 1.15 at step 1000 → stop
- Step time > 350ms sustained post-loop activation → stop (higher budget than 045 given extra passes)

## Cost estimate

| Rung | Arms | Cost |
|---|---|---|
| Mini 4×H100 | 2 parallel arms (~20 min) | ~$14 |
| Official 8×H100 | 1 arm × 3 seeds | ~$18 |
| Total (if one arm wins) | | ~$32 |

## Open questions for interview

1. For Arm B (Loop345): the loop activation log line prints `encoder_indices` and
   `decoder_indices` — verify layer 3 appears in the looped sequence as expected.
2. Does 1/L init for layer 3 interfere with its role as the last "routing" layer before
   the loop? Watch for loss spike at loop activation (step ~2300).
3. Both arms are 17 layer-passes post-loop. If actual tok/s drops below ~2.5M sustained,
   stop — that would indicate unexpected overhead beyond the layer-count prediction.
