# Quant repair — landscape, results, and final state

**Created:** 2026-04-27
**Last updated:** 2026-04-27 (post-046 family results)
**Sources:** technique-timeline (PR survey) + literature survey + 046 family runs (A-H)
**Status:** **DIRECTION ESSENTIALLY CLOSED** — see results section

## TL;DR after 046 family

**Only one real win**: SDClip tightening (-0.00086 to -0.00216 BPB). **All over byte cap.**
**Every other quant repair lever**: null or negative.
**Byte-saving moves to make SDClip wins legal**: all hurt more than they help.

**Practical conclusion**: stuck at the legal cap with 1.07467 quantized baseline. Quant-side direction exhausted on this stack.

## Setup baseline

- Stack (from `045-loop-layer-improvements/armD` checkpoint, used as eval reference):
  - GPTQ int6 matrices / int7 emb
  - SDClip (k=12.85 mat, k=12 MLP, k=13 attn, k=15 emb)
  - LQER asym rank-4 top-3 (covers tok_emb + 11 MLP fc tensors)
  - EMA decay 0.9965, 16-batch GPTQ calibration
  - SpinQuant code wired but disabled (correctly so — see results)
- Quant tax: +0.00920 (post-quant 1.07467 − pre-quant EMA 1.06547)
- Artifact size: 15,953,718 bytes / 16,000,000 cap = **46 KB headroom (0.29% slack)**
- 5-seed σ on quant gap: ~0.0005 (anything <0.001 is noise)

## Cap = 16,000,000 bytes DECIMAL (not 16 MiB)

Verified in upstream code (`records/track_non_record_16mb/2026-03-21_DepthRecurrence_MixedPrecisionQuant/train_gpt.py`: `if total_artifact > 16_000_000`) and PR #1797 body. Common mistake to use 16,777,216 (binary MiB) — that's wrong.

## RESUME_FROM_CKPT infrastructure (committed)

Added in commit `0ea6a97` (later `381baf2` and `44b55b0` for fixes). Lets us iterate on quant settings via:
- Skip train_model()
- Load `final_model.pt` (post-EMA) into base_model
- Set `looping_active=True` (CRITICAL — without this pre-quant eval is broken)
- Run pre-quant eval → serialize (= GPTQ) → deserialize → quantized eval

Cost per arm: ~$1, ~5-7 min on 4×H100 (vs ~$5-10 + 20 min for full screen).

**Reference checkpoint**: `/workspace/runs/045-loop-layer-improvements/armD/final_model.pt` on NE-1 volume.

## RESULTS — 046 family (all run via RESUME_FROM_CKPT)

### 046B-tight (SDClip tightening) — **REAL WIN, ILLEGAL**

| Arm | Quantized | Δ vs verify (1.07467) | Size | Legal? |
|---|---|---|---|---|
| 046B-tight (MLP=11.5/ATTN=12.5/EMBED=14.5) | **1.07381** | **-0.00086** | 16,183,797 | ❌ +184 KB over |
| 046B-loose (MLP=12.5/ATTN=13.5/EMBED=15.5) | 1.07499 | +0.00031 | 15,724,208 | ✓ legal but worse |
| 046B-mlp-only-tight (MLP=11.5 only) | 1.07422 | -0.00045 | 16,095,521 | ❌ +96 KB over |

**Lever direction: tighter wins, looser hurts. Confirmed real signal.**

### 046G-tighter (push the SDClip lever) — **MONOTONIC, ALL ILLEGAL**

| Arm | Quantized | Δ | Size | Legal? |
|---|---|---|---|---|
| 046G-tighter (-1.0σ each) | 1.07322 | -0.00146 | 16,427,706 | ❌ +428 KB |
| **046G-tightest (-1.5σ each)** | **1.07251** | **-0.00216** | 16,688,654 | ❌ +689 KB |
| 046G-mlp-only-tighter | 1.07379 | -0.00088 | 16,254,785 | ❌ +255 KB |
| 046G-attn-only-tighter | 1.07432 | -0.00035 | 16,078,450 | ❌ +78 KB |
| 046G-embed-only-tighter | 1.07440 | -0.00027 | 16,007,406 | ❌ **+7 KB** (closest!) |
| 046G-tight-emb8 | 1.07255 | -0.00212 | 16,713,164 | ❌ +713 KB |

**Findings**:
- SDClip is **monotonic** — every -0.5σ tightening keeps winning (~-0.0007 BPB per notch)
- MLP tightening carries ~50% of the gain but adds ~250 KB per -0.5σ
- ATTN tightening: smaller win, smaller bytes
- EMBED tightening: smallest win (-0.00027), almost legal (+7 KB only)
- Each -0.5σ adds ~250 KB of artifact (less compressible quant weights)
- EMBED_BITS=8 doesn't help over emb=7 with tight clip

### 046A (calibration batches) — null

| Arm | Quantized | Δ |
|---|---|---|
| GPTQ_CALIBRATION_BATCHES=32 | 1.07459 | -0.00008 |
| GPTQ_CALIBRATION_BATCHES=64 | 1.07462 | -0.00005 |
| GPTQ_CALIBRATION_BATCHES=128 | 1.07461 | -0.00006 |

**Saturated at 16 batches.** Confirms PR #756 finding.

### 046C (SpinQuant) — broken / closed

Catastrophic +6.78 BPB on RESUME_FROM_CKPT path. Root cause: residual-stream rotation requires per-channel multiplier folding (attn_scale, mlp_scale, skip_weights, resid_mix) that's NOT IMPLEMENTED. Spec 009 deferred this variant for that reason.

Historical SpinQuant on this stack via `spinquant_hotstart.py`:
- 009/internal_only: 1.06731 vs baseline 1.06728 = **+0.00003 (null)**
- 010/port_1695: 1.06723 vs 1.06728 = **-0.00005 (null)**
- Multiple PR ablations agree (null on Muon weights)

**Even if we fixed the code, the upside is null. Direction permanently closed.**

### 046D (LQER knobs) — null

| Arm | Quantized | Δ | Size delta |
|---|---|---|---|
| LQER_TOP_K=5 | 1.07466 | -0.00001 | +7 KB |
| LQER_TOP_K=8 | 1.07466 | -0.00001 | +14 KB |
| LQER_ASYM_GROUP=32 | 1.07467 | 0 | +2 KB |
| LQER_ASYM_GROUP=128 | 1.07468 | +0.00001 | +1 KB |
| LQER_RANK=6 | 1.07467 | 0 | +5 KB |

**LQER is at its Pareto point.** Adding more bytes to LQER (more rank, more tensors, finer groups) gives ~0 BPB. The first 3 tensors carry ALL the LQER recovery (~-0.009 per PR #1797). Marginal tensors contribute nothing.

Going DOWN in LQER untested but bounded:
- TOP_K=2: ~-3 KB savings, unknown cost
- TOP_K=1: ~-7 KB savings, likely costs -0.005+ (most LQER credit on MLP fc)
- LQER off: ~-30 KB savings, costs -0.009 BPB
- RANK=3: ~-3 KB savings, unknown cost

**LQER byte savings cap out at ~30 KB for full removal — proportional to the BPB lost.**

### 046E (post-quant param fit) — hurts

Fit ~32K passthrough fp16 params (attn_scale, mlp_scale, resid_mix, q_gain, etc.) to match base_model logits via AdamW MSE.

| Arm | Quantized | Δ |
|---|---|---|
| 046E-iters5 (lr=1e-3, 5 iters × 8 batches) | 1.07510 | +0.00043 |
| 046E-iters10 | 1.07516 | +0.00049 |
| 046E-cb16 (16 batches) | 1.07516 | +0.00049 |
| 046E-lowlr (lr=3e-4) | 1.07501 | +0.00034 |

**All arms hurt.** MSE objective doesn't descend cleanly (bounces 0.32-0.35). 32K params don't have capacity to compensate for matrix-weight quant error. **TTT does this job better in the eval pipeline.**

### 046F (AR self-gen calib) — null

| Arm | Quantized | Δ |
|---|---|---|
| 046F-ar-temp1 (temp=1.0, seq_len=512) | 1.07481 | +0.00014 |

Within noise floor. Matches PR #756's prediction (~±0.0003 vs val-data calib). Skipped remaining arms — same null expected.

### 046H (EMBED_BITS reduction) — hurts

| Arm | Quantized | Δ vs verify | Size | Legal? |
|---|---|---|---|---|
| EMBED_BITS=6 pure (clip=15) | 1.08072 | **+0.00605** | 15,431,094 | ✓ legal |
| EMBED_BITS=5 pure | 1.10845 | +0.03378 | 14,943,812 | ✓ legal |
| EMBED_BITS=6 + clip=12 | 1.07803 | +0.00336 | 15,586,422 | ✓ legal |
| EMBED_BITS=5 + clip=12 | 1.09452 | +0.01985 | 15,059,230 | ✓ legal |
| EMBED_BITS=5 + clip=10 | 1.08769 | +0.01302 | 15,196,807 | ✓ legal |
| EMBED_BITS=6 + 046B-tight | 1.07943 | +0.00476 | 15,657,893 | ✓ legal |
| EMBED_BITS=6 + 046G-tighter | 1.07859 | +0.00392 | 15,898,038 | ✓ legal |
| EMBED_BITS=6 + 046G-tightest | 1.07737 | +0.00270 | 16,161,778 | ❌ +162 KB |

**EMBED_BITS reduction direction is dead.** Cost (+0.006 for emb6) is 6× the SDClip win. LQER doesn't absorb int6 emb error like it does for MLP fc.

## Param-cut historical experiments — all hurt

| Run | What was cut | Pre-quant cost vs current baseline |
|---|---|---|
| 041G (mlp_late_mult 4.0 → 3.0) | MLP capacity in late layers | +0.006 BPB |
| 041D (loop_start=5,loop_end=5) | shrunk loop band | +0.0045 BPB |
| 045 armB (MLP_ONLY_FROM_PASS=1) | skip attn on loop passes 2+ | **diverged, killed** |
| 045 armG (LOOP_LR_SCALE=recip) | gradient 1/L on loop layers | early signs of slowing pre-loop learning |

Pattern: **any param removal hurts pre-quant by way more than the freed bytes recover via clip-tightening.** Model is highly tuned; marginal capacity reductions don't recover.

## Key empirical facts (don't forget)

1. **Cap is 16,000,000 bytes DECIMAL** (not 16 MiB binary). Verified.
2. **Rotation methods are null on this stack** (4 ablations agree). SpinQuant lever permanently dead — Muon orthogonalization already produces sub-Gaussian weights, GPTQ act-order handles outliers.
3. **LQER is at its Pareto point**. Adding more rank/coverage gives 0. Removing trades bytes for proportional BPB loss.
4. **SDClip is the only real lever** but hits the byte cap immediately. Each -0.5σ tightening: -0.0007 BPB at +250 KB.
5. **EMBED_BITS lower than 7 hurts more than it helps.** LQER doesn't absorb emb quant error like it absorbs MLP fc error.
6. **Post-quant fitting on small params can't compete with TTT.** TTT does this job better at eval time.
7. **GPTQ calibration is saturated at 16 batches.**
8. **The artifact is ~95% large matrices** (tok_emb + attn + MLP). Everything else is fingernails.

## Where the bytes live

| Component | Approx bytes | % of artifact |
|---|---|---|
| `tok_emb` (int7+LQER) | ~3.7 MB | ~23% |
| 11× attn matrices (int6+GPTQ) | ~5.5 MB | ~34% |
| 11× MLP fc + proj (int6+GPTQ+LQER) | ~6.1 MB | ~38% |
| Passthrough fp16 (q_gain, scales, gates, lambdas) | ~50-100 KB | <1% |
| LQER A+B factors (12 tensors) | ~30 KB | <1% |
| Code (compressed) | ~36 KB | <1% |
| Brotli/wrapper overhead | ~30 KB | <1% |

## Untouched ideas (low EV, but unexplored)

These are the remaining "find more bytes" candidates we haven't tried:

1. **Code dead-path strip** — strip RESUME_FROM_CKPT, 046E fit, 046F AR loader from final submission. ~10-20 KB potential. Cheap, low risk.
2. **Quantize passthrough fp16 → int8** — ~50-100 KB potential. ~30 lines code. Per-channel scales currently fp16; int8 might add noise but mostly fine.
3. **Surgical near-zero weight strip** — Muon WD pushes some weights to ~0. Set hard threshold pre-quant, brotli compresses zeros well. ~50-300 KB potential. Risky.
4. **Single-layer int5 mix** — int5 on ONE least-sensitive layer. ~50-100 KB. Risky (PR #1646 said int5 fails wholesale).
5. **Different compressor** (lzma vs brotli) — speculative ±50 KB.

**Even the best of these (passthrough int8 ~80 KB) only unlocks 046B-tight (-0.00086) legally.** Total potential -0.001 BPB after TTT compression.

## Untouched code-needed ideas (multi-day work)

These would take 2-7 days of engineering with uncertain payoff:

1. **AdaRound** — per-weight rounding decision optimization. Likely overlaps LQER.
2. **OmniQuant** — block-wise reconstruction with learnable clipping. Maxed out at SDClip-level wins probably.
3. **Sequential cross-layer GPTQ** (#1664 design) — code-complete elsewhere, never run on our stack.
4. **Block reconstruction (BRECQ)** — likely overlaps LQER.
5. **Per-tensor Hessian-derived clip** (#1689) — Hessian sensitivity per-tensor instead of per-category.

## Fundamentally new ideas (separate file)

When we exhausted variants of existing techniques, brainstormed five new
paradigms in `research/ideas/quant-repair-fundamentally-new.md`:

- **A. Deploy-time quant repair using eval headroom** (HIGHEST EV) — uses
  PR #1797's unused 100-180s of leaderboard compute; bypasses 16MB cap entirely
- **B. NF4-style non-uniform quantization** — distribution-fit levels at same bytes
- **C. Vector quantization with shared codebook** — different storage paradigm
- **D. Self-distillation during quantization** — multi-layer KL vs per-layer MSE
- **E. Tensor-train decomposition for embedding** — could free 1-2 MB if it works

All un-tested. See dedicated file for detailed analysis and ranking.

## Final practical state

**Best legal config**: armD baseline = 1.07467 quantized. Same as #1797's pre-TTT (1.07443 ± 0.00065).

**Post-TTT projection**: ~1.062, competitive with #1797's 1.06157.

**Quant repair direction effectively closed.** Engineering investment is no longer justified given:
- 3 days to deadline
- All cheap config-only levers tested and either null/negative
- All byte-saving moves cost more BPB than they unlock
- Surviving directions need multi-day code work with uncertain payoff

**Recommended pivots**:
1. Multi-seed final at current config WITH TTT enabled (~$30, the actual submission)
2. Optional: passthrough fp16→int8 quantization (~$1, only untouched cheap byte-saver, unlocks ~-0.0006 post-TTT)
3. EMA decay sweep was queued but user explicitly skipped — they're trying other things in training

## What worked / didn't — summary table

| Lever | Result | Status |
|---|---|---|
| 046B/G SDClip tighter | -0.00086 to -0.00216 BPB, all illegal | proven win, byte-blocked |
| 046A calib batches | null | closed |
| 046C SpinQuant | catastrophic / null | closed |
| 046D LQER knobs | null | closed |
| 046E post-quant fit | hurts | closed |
| 046F AR self-gen calib | null | closed |
| 046H EMBED_BITS reduction | hurts more than helps | closed |
| Passthrough fp16→int8 | untested | open (only viable byte-saver remaining) |
| Surgical zero-weight strip | untested | open (risky) |
| AdaRound / OmniQuant / BRECQ | untested | open (multi-day code) |
