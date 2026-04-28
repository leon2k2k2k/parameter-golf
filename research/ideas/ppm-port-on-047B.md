# PPM-D port to 047B with anti-hijack tuning

**Date:** 2026-04-28
**Frozen as:** `research/specs/052-ppm-port-tuned.md`

## Genesis

PR #1850 (someone114514, 2026-04-26) submitted a PPM-D byte-mixture mechanism for the first time on a vanilla SP8192 base, achieving val_bpb = 1.00495 — a ~0.05 BPB improvement over the prior SOTA. Single-pass PPM scoring fits the 600s eval budget (252s on competition cluster, per their seed log).

PR #1857 (dexhunter, 2026-04-27) tried the same mechanism on a stronger PR #1787-base NN, hit 1.0322, and was closed for chronological-priority loss to 1850. Their report ("OMP chunked: 957s baseline → 95-190s") established the 8-way chunked timing.

Our 047B (KV-shrink CaseOps base, post-quant 1.077) is between the two NN bases in strength. We expected the PPM gain to fall between, and a head-to-head against 1850 to be tight.

## What we measured (2026-04-28)

Full-val per-byte PPM analysis on 047B:
- 86% of PPM's saves come from **space bytes** (byte-bigram patterns NN spreads probability over)
- 34% from **within-doc rare-term recall** (Mississippi, "CC Madhya 9", legal boilerplate — bytes PPM has memorized in-stream)
- PPM HURTS on every non-alpha category and on D7-D10 of NN difficulty (where NN was already right)

The gate-fire rate is INVERTED from where you'd want — 35% on D10 (NN nailing it) vs 7.7% on D1 (NN catastrophic). PPM was firing exactly on the bytes where it could only damage.

## The anti-hijack discovery

Hijack pattern: gate fires HIGH (PPM confident) but NN was ALREADY confident on the actual byte; their top-1 disagrees; mix takes a `−log2(λ_lo) = 4.32` bit hit per occurrence. About 0.34% of bytes; total cost ~2M bits = 8.9% of total PPM wins.

Fix: suppress the gate when NN gave the actual byte high probability. New parameter `nn_skip_thr_nats`: if `nn_logp > -nn_skip_thr`, treat gate as not fired.

Empirical sweep on cached post-quant per-byte data:
- 1D anti-hijack alone (thr=0.9): −0.007 BPB
- 1D lower threshold alone (thr=0.85, no anti-hijack): −0.002 BPB
- **2D combined (thr=0.76, nn_skip_thr=0.40 bits)**: **−0.026 BPB**

The two knobs interact strongly. Lower `thr` fires the gate on more bytes, and anti-hijack catches the new hijacks without losing the new wins. This is a roughly free 0.018 BPB beyond what the single-knob sweep would predict.

## End-to-end validation (Phase 1)

Patched 047B's actual `train_gpt.py` with the PPM block + anti-hijack. Ran EVAL_ONLY=1 on 1×H100 with the existing `final_model.int6.ptz`.

Result:
```
diagnostic quantized val_bpb: 1.07647 (vs published 1.07653, drift 0.00006)
ppm_full_native ... mix_bpb_sidecar = 1.00506
total_pass_time: 184.4s
```

This is the leaderboard-comparable number, no drift correction needed. Matches our cached-NLL prediction (1.00505) within 0.0001 BPB.

## What this gets us

Predicted submission: ties 1850 within seed noise. **Doesn't beat 1850.** A weaker NN (1850's vanilla SP8192) leaves more headroom for PPM, and that headroom gain offsets our marginally stronger NN.

To beat 1850 we need:
- A NN that's ≥0.020 BPB stronger than 047B pre-quant (e.g., 1797's 1.06157 area)
- OR a post-NN lever orthogonal to PPM — TTT is the obvious candidate, since 34% of PPM's mechanism overlaps with TTT but 66% is non-overlapping

## Next move

Spec 052 lands the PPM-port-with-anti-hijack as a clean, validated submission candidate at parity with 1850. After that, Phase 2 explores: (a) PPM + TTT stack on 1797 base, (b) PPM on 050-series once stable, (c) PPM order sweep.
