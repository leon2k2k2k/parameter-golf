# Evaluation — Spec 047 (global NUM_KV_HEADS=2, all 11 layers)

**Run dir:** `runs/047-kv-group4-screen/`  
**Commit:** `e021255` on `exp/045-loop-layer-improvements`  
**Baseline:** spec 045 armD (AC-fix), pre-quant EMA **1.06479**  
**Eval date:** 2026-04-27  
**Status:** killed by user at step 3600 (~80% of budget); no final val_bpb

## Result — killed, clear negative trajectory

| step | AC-fix train_loss | 047 train_loss | Δ |
|---|---|---|---|
| 100 | 3.5927 | 3.5963 | +0.0036 |
| 500 | 2.6635 | 2.6738 | +0.0103 |
| 1000 | 2.7636 | 2.7762 | +0.0126 |
| 1500 | 2.5898 | 2.6038 | +0.0140 |
| 2000 | 2.6492 | 2.6668 | +0.0176 |
| 2500 | 2.4993 | 2.5298 | **+0.0305** ← loop activated ~step 2400 |
| 3000 | 2.5657 | 2.5962 | **+0.0305** |
| 3500 | 2.3863 | 2.4228 | **+0.0365** ← trending up |

At step 3500, Δ train_loss = +0.0365 and still widening. Extrapolating to ~step 4500 final, likely +0.04–0.05 in train_loss — which would translate to roughly +0.010–0.015 bpb pre-quant at the end. Clearly above the kill gate (>1.06979).

**Verdict: KILL.** Terminated at step 3600 to preserve pod budget. Signal was unambiguous.

## Pattern: loop amplifies the GQA penalty

The Δ was ~+0.013 pre-loop-activation, then jumped to ~+0.030 immediately at loop activation (~step 2400) and continued widening. This is the "loop multiplier" effect: KV heads run 3× per forward in the loop band, so any GQA capacity loss is applied 3 times per example in those layers. Global KV=2 affects all 11 layers including the 8 non-loop layers that don't have this multiplier mitigation.

## Contrast with 047B (loop-layer-only KV=2)

| | 047 (global) | 047B (loop-only) |
|---|---|---|
| Layers affected | 11 of 11 | 3 of 11 |
| Final pre-quant bpb | ~+0.010–0.015 (est.) | **+0.00284** |
| Verdict | KILL | iterate |
| Train_loss Δ at loop activation | +0.030 | +0.021 |
| Train_loss Δ at step 3500 | +0.037 (widening) | +0.033 (recovering via EMA) |

The difference is stark. 047B's end-of-run +0.0028 bpb vs 047's projected +0.010–0.015 bpb shows that the non-loop layers are much more sensitive to KV head reduction. This makes sense: the non-loop layers run once per forward, have no redundancy from multiple passes, and some (early layers = feature extraction, final layers = logit projection) have no analogue in a depth-recurrent stack.

## Decision — KILL; 047B approach more promising

1. **Global KV=2: KILL.** Loss degradation too large, widening trend gives no hope of recovery.
2. **Lesson: loop layers specifically tolerate KV compression.** The 047B result (+0.0028 for 9 effective attention ops) is real. Non-loop layers don't.
3. **Path forward:** fix the 047B size issue (shrink `kv_bank` shape to exclude loop rows, not add `loop_kv_bank` on top) and combine with a quality-recovering lever (047D AdaLN, 047C LoRA).
4. **KV=3 mid-point not worth trying.** If KV=2 globally is +0.010–0.015, KV=3 would be half that, but still likely +0.005–0.008 on all 11 layers. Not competitive.

## Cost

~$2.45 (4×H100, ~35 min wall including prewarm). No checkpoints emitted (killed before GPTQ).  
Inductor cache stashed at `/workspace/.inductor_cache_e021255_kv2` (partially — rsync was interrupted by pod stop; remove before any 047 rerun).

## Cross-references

- Spec: `research/specs/047-kv-group4-screen.md`
- Contrast: `research/evaluations/047B-loop-kv-shrink-screen.md` (loop-only variant)
- Next: fix 047B size bug, combine with 047D
