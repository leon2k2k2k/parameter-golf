# Evaluation 041B — original loop (layers 3-5), NUM_LOOPS=1, frac=0.59

**Spec:** `research/specs/041B-original-loop-1pass-late-activation-screen.md`
**Status:** COMPLETE (wallclock cap, step 6002/20000)

## Result

| Metric | Value |
|--------|-------|
| pre-quant EMA val_bpb | **1.06842** |
| Δ vs baseline (1.06514) | +0.00328 *(worse)* |
| Δ vs 040 no-loop ablation | — *(no 040 val_bpb available)* |

**Verdict: KILL** — 1.06842 ≥ 1.0670 threshold. Full loop range (layers 3-5) at 1-pass depth with late activation (frac=0.59) does not improve over baseline.

## Run details

- Pod: `ecbzpf0mej3x1w` (US-NE-1, 4×H100)
- Branch: `exp/039b-loop-band-activation-screen`, commit `5bbf12f`
- Loop activated: step 3862 (expected ~3878 ✅)
- Final step: 6002 (expected ~5995 ✅)
- Phase 1 tok/s: ~4292K | Phase 2 tok/s: ~4080K (1-pass loop, faster than 2-pass baseline's ~3200K)

## Loop activation

Loop activated cleanly at step 3862 (frac=0.460 elapsed wall fraction — note: frac target was 0.59 of wallclock but activation landed slightly early due to step-count scheduling). No anomalous spike observed.

## Step-matched train loss

| Step | Baseline | 040 (no-loop) | 041B | Δ 041B vs baseline | Δ 041B vs 040 |
|------|----------|---------------|------|--------------------|---------------|
| 3500 | 2.4059 | 2.4527 | 2.4540 | +0.048 | +0.001 |
| 3800 | 2.5215 | 2.5829 | 2.5873 | +0.066 | +0.004 |
| 4000 | 2.4050 | 2.4684 | 2.4609 | +0.056 | −0.008 |
| 4100 | 2.3630 | 2.4317 | 2.4210 | +0.058 | −0.011 |
| 4400 | 2.3718 | 2.4520 | 2.4331 | +0.061 | −0.019 |
| 4800 | 2.2884 | 2.4025 | 2.3735 | +0.085 | −0.029 |
| 5100 | 2.3130 | 2.4211 | 2.3831 | +0.070 | −0.038 |
| 5500 | — | 2.4626 | 2.4166 | — | −0.046 |
| 5800 | — | 2.3714 | 2.3317 | — | −0.040 |
| 6000 | — | 2.3822 | 2.3522 | — | −0.030 |

## Analysis

041B consistently beats 040 (no-loop ablation) by a widening margin post-activation (~0.01 at step 4100, ~0.046 at step 5500), confirming the 1-pass loop on layers 3-5 does carry a real training-loss benefit over no recurrence.

However, it cannot close to baseline. The gap vs baseline at the last comparable step (5100) is −0.070 — larger than earlier in the run and not converging. The 1-pass loop at this layer range gives weaker recurrent capacity than the 2-pass baseline (14 effective passes vs 17), which may explain the persistent gap.

The frac=0.59 activation point also landed at step 3862 — earlier than intended if the correction note in 041Bb is right (actual loop throughput is ~3277K tok/s, not ~3385K, giving fewer post-loop steps than expected). 041Bb re-runs this at frac=0.65 to deliver the correct ~6000 steps with accurate phase budgeting.

## Conclusion

041B = 040 + 1-pass loop benefit, but still well below baseline. The 1-pass loop on full layer range beats no-loop but is insufficient at this depth. The corrected frac (041Bb) is worth running to ensure the step budget wasn't the confound, but the gap to baseline (+0.00328) is large enough that 041Bb is unlikely to reverse the verdict.

## 041 family summary (so far)

| Arm | Loop config | frac | pre-quant BPB | Δ vs baseline | Verdict |
|-----|-------------|------|--------------|---------------|---------|
| baseline | 2-pass, layers 3-5 | 0.35 | 1.06514 | — | — |
| 040 (no-loop) | none | — | — | — | ablation only |
| **041B** | 1-pass, layers 3-5 | 0.59 | **1.06842** | **+0.00328** | **KILL** |
| 041Bb | 1-pass, layers 3-5 | 0.65 | pending | — | — |
| 041A | 2-pass, shrunk range | 0.59 | pending | — | — |
| 041C | 1-pass, shrunk range | 0.59 | pending | — | — |
