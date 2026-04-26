# Spec 046A — GPTQ calibration batches sweep

**Slug:** `046A-calib-batches-sweep`
**Created:** 2026-04-27
**Status:** READY (after 046 PASS)
**Branch:** `exp/046-quant-repair`
**Commit:** `0ea6a97`
**Parent:** spec 046 (verification), idea Q5 in `research/ideas/quant-repair.md`

## Hypothesis

Our baseline uses `GPTQ_CALIBRATION_BATCHES=16`, which is **1/8 the literature
standard of 128** (GPTQ paper, Frantar et al.). Hubara et al. 2021 established
that calibration quality + size matters; saturation is around 128-256. PR #756
ablation confirmed self-gen calib closes 84% of the val/random gap on this
competition's stack — meaning calib data quality is a real lever here.

If 16 batches is leaving signal on the table, more batches → cleaner per-row
Hessian → tighter quant.

## Predicted outcome

Per literature: −0.001 to −0.003 BPB on quant tax. Expect diminishing returns
above 64. Realistic best-case ~1.0742 quantized (−0.0005 from baseline).

## Arms

| Arm | `GPTQ_CALIBRATION_BATCHES` | Expected vs verification baseline |
|---|---|---|
| 046A-32 | 32 | small −0.0003 to −0.001 |
| 046A-64 | 64 | −0.001 to −0.002 |
| 046A-128 | 128 | −0.001 to −0.003 (target) |

Run all three sequentially on the same pod after 046 PASS.

## Config diff (vs 046 verification baseline)

```bash
GPTQ_CALIBRATION_BATCHES=32   # arm 046A-32
GPTQ_CALIBRATION_BATCHES=64   # arm 046A-64
GPTQ_CALIBRATION_BATCHES=128  # arm 046A-128
```

All other env vars **identical to 046 verification**. RESUME_FROM_CKPT same path.
Set `RUN_ID` per arm: `046A-calib-batches-32` etc.

## Acceptance

Reference = 046 verification quantized number (expected ~1.07467).

Per arm:
- **Win**: < 1.0735 (−0.0012 from reference)
- **Strong win**: < 1.0730 (−0.0017)
- **Noise**: 1.0735 – 1.0760
- **Kill**: > 1.0760 (worse)

If 046A-128 wins clearly and 046A-64 is between → run 046A-256 for diminishing-returns probe.

## Cost

~$3 total for 3 arms (~5-7 min each).

## What to watch

- `GPTQ:collected N Hessians in Xs` line — verify N matches expected count
- Quantized eval val_bpb — the headline
- GPTQ wallclock per arm — should grow ~linear in batch count

## Decision

| Result | Next |
|---|---|
| Any arm wins | Adopt as new default; fold into 046B+ runs |
| All noise | Calib direction closed; defer batch-count tuning |
| 128 worse than 32 | Check pod variance / GPTQ stochastic issue first; if persists, 16 was actually well-tuned |
