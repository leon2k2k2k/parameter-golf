# Idea — Negative slope annealing: start at √0.5, converge to 0.5

## One-line

Train with `NEGATIVE_SLOPE=√0.5≈0.707` early to get aggressive gradient kick,
then anneal back down to `0.5` so the model can converge cleanly at warmdown.

## Background — how we found this

The fused `LeakyReLU(s)²` Triton backward had a bug from commit `207f38e` to
`5bbf12f`: it computed `2·s·x` instead of the correct `2·s²·x` for negative
inputs. With `NEGATIVE_SLOPE=0.5`, the buggy backward used an effective slope
of `√0.5 ≈ 0.707` — forward and backward were inconsistent.

This accidentally created a curriculum:
- **Forward**: `leaky_relu(x, 0.5)²` — same as normal
- **Backward**: gradient matching `leaky_relu(x, √0.5)²` — more permissive,
  2× larger negative-side gradient

What we observed in the contaminated runs (039b, 042):
- The first 10 minutes of a 4×H100/20min run showed a **dramatic boost** vs
  clean baseline, especially around and after loop activation
- After 10 minutes the model **stopped improving and vibrated** — the
  forward/backward inconsistency prevented convergence at warmdown
- At the 10-minute mark the contaminated model had already reached its best
  state; the second half was wasted

The clean 039b re-run (fixed backward, s=0.5) confirmed: the model converges
smoothly all the way to step 5000 with no vibration, landing at BPB 1.06514.

## The hypothesis

The bug accidentally gave us something useful: **an aggressive early phase
followed by an inability to exploit it**. The fix is to do this intentionally:

1. Start training with `s = √0.5 ≈ 0.707` — more gradient flow through
   negative activations, faster early adaptation, especially beneficial when
   the recurrent loop fires for the first time
2. At some point (e.g. when loop activates, or at warmdown start), **anneal
   `s` back down toward `0.5`** — forward and backward become consistent,
   model can settle cleanly

This is slope curriculum / annealing. The kick is real; the wobble is
unnecessary. Separate them.

## Update — 2026-04-25: early loss drop is 86% LR effect, not recurrence

From the 039bE vs 039bG comparison:

- 039bG (LR floor, loop@0.35) beat baseline by −0.123 train loss at step 2800
- 039bE (LR floor, loop@0.175) beat baseline by −0.144 — only 0.020 more
- That 0.020 cost ~1.9min wallclock penalty from slower post-loop throughput
- **Conclusion**: early loop onset is a net negative under wallclock cap; loop@0.35 is the right base for LR schedule experiments. Don't spec loop@0.175 unless the hypothesis is specifically about loop timing.

## Update — 2026-04-25: clean loop@0.175 already beats the contaminated run

039bE (clean code, `NEGATIVE_SLOPE=0.5` throughout, `loop@0.175`) was run on 4×H100/1200s.
At step 2500 — the last common point before the contaminated 039bA run ended — the comparison is:

| Arm | Train loss @ step 2500 |
|---|---|
| contaminated 039bA (√0.5 bwd + ptanh, loop@0.35, 600s) | 2.3981 |
| **039bE (clean, s=0.5, loop@0.175, 1200s)** | **2.3457** |
| clean baseline (s=0.5, loop@0.35, 1200s) | 2.5104 |

039bE has the best mid-training signal of any run ever — cleaner and lower than the
contaminated run we previously thought was exciting. And it uses no tricks: same
`s=0.5`, just earlier loop onset.

The problem: 039bE's final val_bpb (1.06916) is **worse** than baseline (1.06514).
Earlier loop onset means the model spends more wallclock in the slow post-loop regime,
leaving fewer effective warmdown steps in the same 1200s budget. The mid-training
advantage is real but doesn't convert to a final win yet.

This sharpens the hypothesis: we don't need the √0.5 kick for the early advantage —
`loop@0.175` already gives it. What we need is a way to **cash in that head start at
warmdown**. Slope annealing is one candidate; retuning the LR schedule for earlier loop
onset is another.

## Why the kick matters at loop onset

With `loop@0.35`, the loop activates at ~step 2280. The recurrent layers need
to adapt quickly from a feed-forward initialization. A larger negative slope
means bigger gradient steps through negative activations in those first few
hundred post-loop steps — effectively a faster adaptation phase for the
newly-activated recurrent stack. Then annealing back to 0.5 during warmdown
lets the converged recurrent state settle.

Pairs naturally with `loop@0.175` (earlier loop onset), since there is more
post-loop budget to exploit the kick before annealing back down.

## Implementation options

The fused Triton kernel uses `NEGATIVE_SLOPE: tl.constexpr` (compiled in).
Changing it mid-run requires one of:

1. **Two-phase recompile**: run at s=0.707 until transition step, then
   relaunch the kernel at s=0.5 (Triton recompiles on next call with new
   constexpr — one-time pause of ~seconds)
2. **Eager path for transition**: use eager MLP during the annealing window
   (no recompile needed, slight perf hit for that window only)
3. **Non-constexpr slope**: pass slope as a runtime tensor argument (small
   ongoing perf hit, cleanest for smooth annealing)

Option 1 is probably the right first version — simple step function at warmdown
start, minimal code change.

## First experiment to run

Simplest version: **step function at warmdown start**

- Phase 1 (steps 0 → warmdown): `NEGATIVE_SLOPE=0.707`
- Phase 2 (warmdown → end): `NEGATIVE_SLOPE=0.5`

Compare against clean baseline (s=0.5 throughout) on 4×H100, 1200s.

If the step function works, follow up with:
- Smooth cosine annealing from 0.707 → 0.5
- Different transition points (at loop activation vs at warmdown)
- Combining with `loop@0.175`

## Key open questions

- Does the kick survive to final BPB (post-quant, post-TTT), or is it only a
  training-loss effect?
- What is the optimal starting slope? √0.5 is where the bug landed but 0.6,
  0.65, 0.75 are all worth considering.
- What is the optimal transition point?
- Does the annealing rate matter, or is a step function good enough?
