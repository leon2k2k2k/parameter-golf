# Evaluation 039bN — non-uniform MLP width banding (040C: 4.0/5.0/3.4)

**Spec:** `research/specs/039bN-mlp-band-activation-screen.md`
**Status:** COMPLETE (wallclock cap, step 5062/20000; GPTQ crashed post-training)

## Result

| Metric | Value |
|--------|-------|
| pre-quant EMA val_bpb | **1.06704** |
| quantized val_bpb | N/A *(GPTQ IndexError — see below)* |
| Δ pre-quant vs baseline (1.06514) | +0.00190 *(worse)* |
| Δ pre-quant vs 039bL (1.06594) | +0.00110 *(worse)* |

**Verdict: KILL** — worse than both baseline and 039bL. MLP banding (040C) did not improve convergence.

## GPTQ crash note

Serialization crashed after training completed:
```
IndexError: index 0 is out of bounds for dimension 1 with size 0
  gptq_quantize_weight → w_col = W_block[:, j]
```
Likely a GPTQ calibration edge case with the non-uniform hidden dims (2048/2560/1744). The
pre-quant EMA val_bpb was logged cleanly before serialization began; that is the primary metric.

## Loop activation

The wider 2560d middle layers (MLP_MIDDLE_MULT=5.0) produced a notably clean loop activation.
At step 2300, 039bN showed essentially no spike vs baseline:

| Arm | Step 2300 train_loss | Δ vs baseline |
|-----|---------------------|---------------|
| baseline | 2.7198 | — |
| 039bL | 2.9334 | +0.214 *(spike)* |
| **039bN** | **2.6400** | **−0.080** *(no spike)* |

The wider loop core dampened activation shock more effectively than the curriculum approach in
039bM (which got −0.010 via 1-loop start). Despite cleaner activation, this advantage did not
survive the floor_then_linear LR erosion.

## Step-matched train loss

| Step | Baseline | 039bL | 039bN | Δ bN vs baseline | Notes |
|------|----------|-------|-------|-----------------|-------|
| 1400 | 2.5973 | 2.5589 | 2.5644 | −0.033 | |
| 2000 | 2.6552 | 2.5894 | 2.5938 | −0.061 | |
| 2300 | 2.7198 | 2.9334 | 2.6400 | −0.080 | **no spike** |
| 2500 | 2.5104 | 2.4366 | 2.4403 | −0.067 | |
| 3000 | 2.5784 | 2.5182 | 2.5195 | −0.059 | |
| 4000 | 2.4050 | 2.3776 | 2.3753 | −0.030 | eroding |
| 4500 | 2.3854 | 2.3844 | 2.3816 | −0.004 | nearly flat |
| 5000 | 2.4015 | 2.4212 | 2.4138 | +0.012 | behind baseline |

## Analysis

039bN replicated the floor_then_linear pre-loop advantage (−0.061 to −0.080 at steps 2000–2500),
but eroded on the same trajectory as 039bL through the second half. By step 4500 the gap
to baseline was essentially zero (−0.004), and by step 5000 039bN trailed baseline slightly.

The MLP banding effect on loop activation is real and larger than any prior mechanism: at step
2300 the Δ was −0.080 vs baseline (no spike at all), compared to 039bL's +0.214 spike and even
039bM's curriculum-assisted −0.010. The wider loop core absorbs the weight initialization
discontinuity at activation. However, this mechanical advantage is irrelevant to the binding
constraint: floor_then_linear LR starvation erases any pre-loop gain by step 4500.

The GPTQ crash with non-uniform dims is a side-effect of the banding config; the serialization
code doesn't expect varying hidden dims in the GPTQ calibration path.

## Conclusion

039bN = 039bL + cleaner loop activation. The banding provides the smoothest activation of any
arm in this series by a wide margin, but adds +0.00110 BPB vs 039bL rather than subtracting.
The floor_then_linear schedule shape remains the binding constraint for all 039b* arms.

## Full series summary (039bG through 039bN)

| Arm | Schedule | Carry | Loop config | pre-quant BPB | Δ vs baseline |
|-----|----------|-------|-------------|--------------|---------------|
| baseline | WSD | enabled | frac=0.35, 2-loop | 1.06514 | — |
| 039bG | first_half_floor | enabled | frac=0.35, 2-loop | 1.06800 | +0.00286 |
| 039bI | floor_then_wsd | enabled | frac=0.35, 2-loop | 1.06622 | +0.00108 |
| 039bJ | floor_then_linear | enabled | frac=0.35, 2-loop | 1.06598 | +0.00084 |
| 039bK | WSD | disabled | frac=0.35, 2-loop | killed (neutral) | ~0 |
| 039bL | floor_then_linear | disabled | frac=0.35, 2-loop | 1.06594 | +0.00080 |
| 039bM | floor_then_linear | disabled | frac=0.35 (1-loop) → frac=0.65 (2-loop) | 1.06646 | +0.00132 |
| **039bN** | floor_then_linear | disabled | frac=0.35, 2-loop + MLP 4.0/5.0/3.4 | **1.06704** | **+0.00190** |

No LR schedule or MLP banding variant in this series beats baseline. Standard WSD remains optimal.
The loop activation spike (039bL: +0.214) is real and mechanically suppressible (039bN: −0.080),
but suppressing the spike does not rescue late-run convergence under floor_then_linear.
