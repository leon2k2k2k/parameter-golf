# Evaluation: 041H — loop layers 4-5, NUM_LOOPS=2, frac=0.35

**Date:** 2026-04-26
**Status:** NOISE ZONE (borderline kill)
**Spec:** `research/specs/041H-loop45-frac035-screen.md`
**Branch:** `exp/039b-loop-band-activation-screen` @ `5bbf12f`

## Result

| Metric | Value |
|--------|-------|
| Final step | 5534 |
| Loop activated | step 2299 (frac=0.35 ✅) |
| Recurrent steps | ~3235 |
| Pre-quant EMA val_bpb | **1.06693** |
| Peak VRAM | 37,444 MiB allocated / 42,690 MiB reserved |

Note: TRAINING_ONLY_SCREEN=1 — quantized val_bpb not available.

## Comparison

| Run | Steps | Loop start | Recurrent steps | Pre-quant EMA val_bpb | Δ vs baseline |
|-----|-------|-----------|-----------------|----------------------|---------------|
| Baseline (039b) | 5156 | step ~2300 (frac=0.35) | ~2867 | 1.06514 | — |
| 041A (4-5, NL=2, frac=0.46) | 5722 | step 3030 | ~2692 | 1.06545 | +0.00031 |
| **041H (4-5, NL=2, frac=0.35)** | **5534** | **step 2299** | **~3235** | **1.06693** | **+0.00179** |
| 041D (5 only, NL=2, frac=0.46) | 6057 | step 3011 | ~3046 | 1.06993 | +0.00479 |
| 041F (10L, 4-5, NL=2, frac=0.46) | 6137 | step 3286 | ~2851 | 1.07203 | +0.00689 |
| 041G (11L shrunk, 4-5, NL=2, frac=0.46) | 5793 | step 3102 | ~2691 | 1.07180 | +0.00666 |

## Training loss trajectory (loop phase)

| Step | Baseline | 041A | 041H | Δ(H-Base) |
|------|----------|------|------|-----------|
| 2500 | 2.5104 | 2.5252 | 2.5174 | +0.007 |
| 3000 | 2.5784 | 2.6151 | 2.5907 | +0.012 |
| 3500 | 2.4059 | 2.4330 | 2.4267 | +0.021 |
| 4000 | 2.4050 | 2.4387 | 2.4283 | +0.023 |
| 4500 | 2.3854 | 2.4323 | 2.4223 | +0.037 |
| 5000 | 2.4015 | 2.4462 | 2.4325 | +0.031 |
| 5100 | 2.3130 | 2.3527 | 2.3371 | +0.024 |
| 5500 | — | 2.3940 | 2.3922 | — |

## Analysis

041H tested the hypothesis that shrinking loop width (3 layers → 2 layers: LOOP_START=4 instead of 3) at the same activation timing (frac=0.35) as the baseline would yield +loop steps with no capacity penalty.

**Throughput matched spec:** loop phase ran at ~3630–4198K tok/s (vs baseline's ~3717K at the same frac=0.35). The N=15 config (2 loop layers) produced slightly faster loop-phase throughput, yielding ~3235 recurrent steps vs baseline's ~2867 — **+368 loop steps**.

**But val_bpb went the wrong way:** 1.06693 vs baseline's 1.06514 (+0.00179). The prediction was ~1.06437 — we overshot by ~0.00256.

**Key insight from train loss:** 041H tracks between Baseline and 041A throughout the loop phase, not better than baseline. The training loss is ~0.02–0.04 above baseline at matched steps. The removal of layer 3 from the loop (LOOP_START=4 vs 3) appears to cost something in val_bpb, even though RECUR_ALPHA_ENABLED=0 means there's no shape mismatch penalty.

**The 2.2e-6/step extrapolation failed:** We extrapolated from the 041 arc's ~2.2e-6 val_bpb/loop-step rate, which was measured at ≤2867 loop steps. At 3235 steps (extrapolation territory) the model may face diminishing returns, or the 2-layer loop (vs 3-layer baseline) has lower quality-per-step.

**Note on compilation overhead:** Fresh pod with cold inductor cache ate ~5 min of the 20-min budget, reducing effective steps from predicted ~5558 to actual 5534 — a small but real shortfall.

## Verdict

**NOISE ZONE / borderline kill** — 1.06693 is technically above the 1.0670 kill threshold but within noise. The approach is not demonstrably better than baseline. The pure throughput play at frac=0.35 with 2-layer loop does not recover the baseline quality gap.

The frac curve experiments (041J at 0.30, 041I at 0.25) will test whether earlier activation — giving more loop-phase seconds — can overcome the deficit. If the per-step quality at 2+ loop layers matches the 3-layer baseline, earlier activation is the lever.
