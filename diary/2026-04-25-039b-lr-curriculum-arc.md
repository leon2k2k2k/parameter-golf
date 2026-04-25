# 2026-04-25 — The leaky fuckup, the LR-drop hypothesis, and everything we tried

## The bug

When we moved to the 039 code line (commit `207f38e`), we unknowingly introduced
a mismatch in the fused `LeakyReLU(s)²` Triton kernel. The forward was correct —
`f(x) = leaky_relu(x, 0.5)² = 0.25x²` for x < 0. The backward was wrong:

```python
# buggy — what we shipped
c0 = c0 * tl.where(pre0 > 0, 2.0 * pre0, 2.0 * NEGATIVE_SLOPE * pre0)

# correct
c0 = c0 * tl.where(pre0 > 0, 2.0 * pre0, 2.0 * NEGATIVE_SLOPE * NEGATIVE_SLOPE * pre0)
```

The buggy backward computes `2·s·x` for negative inputs, but the correct
derivative of `leaky_relu(x, s)²` is `2·s²·x`. For `s=0.5` this doubles the
negative-side gradient: the model was effectively training with a backward that
matched `s=√0.5 ≈ 0.707` while the forward used `s=0.5`.

In practice this looks fine early in training — large LR swamps the inconsistency.
But during warmdown, when LR collapses and the model tries to settle, the
mismatched curvature creates persistent vibration. The optimizer chases a gradient
that describes the wrong loss surface. Runs on this code line showed noisier EMA
tails and slightly worse final val_bpb than expected.

Every `039b*`, `041*`, `042*`, `043*` run was contaminated. All moved to
`runs/_contaminated_pre_5bbf12f/`. The fix landed at commit `5bbf12f`.

Clean reference is spec 038A (`c8620b6` or earlier) which hardcoded `0.5 * pre`
— the correct formula for `s=0.5` without generalizing to the broken parametric
form. Pre-quant `1.06541920`, post-TTT `1.06287`.

## The observation that launched the LR arc

When we reran the 039b baseline cleanly (commit `0516020`), we noticed something:
the training loss dropped sharply in the first 3.5 minutes, before loop activation
at `ENABLE_LOOPING_AT=0.35` (7min). This drop was entirely explained by the LR
schedule — the standard WSD falls off fast in the first half of warmdown.

The loop activation at 7min added about 1/3 more compute per token but also reset
LR effectively. The question became: **can we exploit the early LR drop? Can we
make the early fast descent do real work before loop activation, and then let the
loop kick in while there's still LR to spend?**

039bG was the first test — `ENABLE_LOOPING_AT=0.175` (3.5min) with
`LR_REWARM_AT=0.175`. Move loop activation earlier so the model has a full half
of the run with loops. Result: pre-quant `1.06800` (+0.00286 vs baseline). The
early boost didn't survive. Faster LR depletion in the first half left less LR
budget for the second half to converge on.

## Everything we tried

### LR schedule variants (all stacked on the 039 baseline)

**039bG** — `ENABLE_LOOPING_AT=0.175`, loop kicks in at 3.5min instead of 7min  
Result: `1.06800` (+0.00286). **Killed.** More tokens per step earlier, but LR
was already low by convergence — spent the budget in the wrong place.

**039bH** — `first_half_floor` schedule: LR floor at 0.35 until 7min (preventing
early drop), standard WSD after. Loop at 3.5min.  
Implementation: a new `LR_SCHEDULE_MODE=first_half_floor` that clamps LR at the
WSD floor value `(0.5 - frac) / warmdown_frac` during the first half.  
Had a lurking complexity: loop activates at 3.5min while LR rewarms at 7min —
two destabilizing events out of phase.  
Result: better mid-training (−0.049 at step 1300 vs baseline) but
converged back to baseline by end.

**039bI** — `floor_then_wsd`: same flat floor until 7min, then jumps back to the
full WSD curve (producing a +0.003 LR spike at the switch point). Loop at 7min.  
Idea: the spike acts as a micro-rewarm at loop activation, letting the model
adapt to the recurrent stack with more LR.  
Result: loss spiked to +0.006 at step 3000, recovered to +0.001 by step 3200,
but couldn't close the gap. Still finished above baseline.

**039bJ** — `floor_then_linear`: same flat floor until 7min, then linearly
descends from the floor value (0.400) to MIN_LR (0.100). No spike.  
Smoother than floor_then_wsd, avoids the destabilization.  
Result: −0.052 at step 3100, faded to +0.023 by step 4900. **Killed.** Same
story — early advantage bleeds away in the second half.

**039bK** — baseline + `RECUR_ALPHA_ENABLED=0` (no frozen carry). Ablation of
the frozen [1.5610, 1.8531, 2.1320] carry scaling inherited from spec 037.  
Ran it to see if frozen carry was a drag. Ended up being the wrong combination
(not stacked on 039bJ as intended), so it was superseded by 039bL.

**039bL** — `floor_then_linear` + `RECUR_ALPHA_ENABLED=0`. The correct combo.  
The idea was that the frozen carry values, learned under a different training
regime and frozen for all subsequent runs, might be mismatched. Removing them
gives a neutral carry.  
Result: strong mid-training (−0.037 to −0.050 at steps 3300–3600) but converged
back to baseline. The pattern was becoming clear.

### The structural lesson from the LR arc

All four LR schedule variants shared the same failure mode: the early advantage
(large Δ at mid-training) always eroded by the final eval. The baseline WSD's
slow, sustained decay through the entire second half is hard to beat in a fixed
20min wallclock. Any schedule that front-loads LR expenditure leaves the second
half underpowered.

The "fast early drop as a booster" hypothesis is false, or at least not exploitable
with LR scheduling alone. The model needs the LR budget to be available when it
enters the convergence phase, not before.

### Structural kick experiments (different axis)

Given that the LR angle is exhausted, we pivoted to structural changes that could
give a late kick without requiring more LR budget.

**039bM** — loop depth curriculum (`LOOP_DEPTH_UPGRADE_AT=0.65`).  
Run starts at 1 loop iteration (ENABLE_LOOPING_AT=0.35 as usual), upgrades to 2
loops at 65% wallclock (13min). The second recurrent pass fires into an already
partially-settled model during the convergence phase. Stacked on 039bL.  
Status: pending.

**039bN** — non-uniform MLP width banding, 040C configuration.  
A prior 2×H100 quick screen (runs 040A/B/C) tested three ways to widen the loop
core (layers 3,4,5 to ×5.0) while staying budget-neutral:

- 040A: shrink both sides equally (×3.625/5.0/3.625) — throughput penalty, underperformed
- 040B: shrink early, keep late (×3.0/5.0/4.0) — mild learning regression  
- 040C: keep early, shrink late (×4.0/5.0/3.4) — stayed on baseline pace

040C was the survivor: early feature-extraction layers (0–2) keep full width,
loop core (3–5) gets widened, late layers (6–10) pay the cost. Width-units sum to
44.0 — exactly budget-neutral vs uniform ×4.0.

Implemented via `MLP_SCHEDULE_ENABLED=1`, bank-slice architecture: the bank
stores `max(hidden_dims)=2560` rows per layer, `_bank_weights(i)` slices `[:hd]`
so each layer only touches its active hidden size. No extra memory overhead beyond
the bank padding. Stacked on 039bL. Leaky backward fix verified intact at `92886ff`.

Status: pending.

## Where we stand

| Arm | pre-quant BPB | Δ vs baseline | Status |
|---|---|---|---|
| baseline (039) | 1.06514 | — | reference |
| 039bG (loop@3.5min) | 1.06800 | +0.00286 | killed |
| 039bH (first_half_floor, loop@3.5) | pending | — | ran |
| 039bI (floor_then_wsd, loop@7) | pending | — | ran |
| 039bJ (floor_then_linear, loop@7) | pending | — | ran |
| 039bK (baseline + no carry) | pending | — | ran |
| 039bL (floor_then_linear + no carry) | pending | — | ran |
| 039bM (loop depth curriculum @65%) | — | — | pending |
| 039bN (040C MLP banding) | — | — | pending |

Win threshold: pre-quant < 1.0641. Kill threshold: ≥ 1.0660.

The LR angle is done. 039bM and 039bN are the remaining live bets.
