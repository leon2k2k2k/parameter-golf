# Quant repair — landscape & ranked ideas

**Created:** 2026-04-27
**Sources:** technique-timeline (PR survey) + literature survey, both 2026-04-27
**Status:** ideas pool, no specs frozen yet

## Context

- Baseline #1736 quant tax (post-quant − pre-quant EMA): **+0.009 BPB**
- Stack: GPTQ int6 matrices / int7 emb, SDClip (k=12.85 mat, k=15 emb), LQER asym rank-4 top-3, EMA decay 0.9965, 16-batch calibration, **SpinQuant code wired but disabled**
- Artifact-cap headroom: ~46 KB under 16 MB
- Reference 5-seed σ on quant gap: ~0.0005 (anything <0.001 is noise)

## Key empirical fact (don't forget)

**Rotation methods are essentially null on this stack.** Four independent ablations agree:
- #670 (abaybektursun): SpinQuant on 11L = −0.0002 BPB
- #1224 (vermissa0ss): per-layer Hadamard = −0.00006 BPB
- #1718 (himanshudongre): "null on Muon sub-Gaussian weights"
- #1695 (X-Abhishek-X): SpinQuant V1 ported, quant tax unchanged at +0.012

Mechanism: Muon's orthogonalization already produces sub-Gaussian weights; GPTQ act-order + Cholesky already handles residual outliers; rotations have nothing to do. **This is why SpinQuant is wired but disabled in #1736.**

→ Drop QuaRot, SpinQuant, DuQuant, FlatQuant, Hadamard pre-rotation from the literature recommendations regardless of how strong they look on LLaMA. Wrong paradigm for this stack.

## What's already in the stack

- **SDClip** (PR #1394) — `c = k·σ(row)`. Foundational primitive every winning PR since stacks on
- **Hessian-aware SDClip** (PR #1412) — modulate clip by row sensitivity, λ=0.175
- **Per-layer adaptive clip** (PR #1586) — MLP=12.0, attn=13.0, **emb=15.0 with int7** → 1.07493 (pre-CaseOps)
- **LQER asym rank-4 top-3** (PR #1797) — recovers ~0.009 of the ~0.012 raw quant tax

## What QAT taught us (and why it's not in stack)

QAT dominated Phase 0 (March), removed in #1218 with note: "appeared to provide little or no benefit." Critical follow-up finding (#1773): **QAT FakeQuantize formula must match save-time clip exactly** — naive absmax mismatch = +0.17 BPB damage. Don't revisit QAT unless we want to align it with SDClip explicitly.

## Ranked ideas

### Tier 1: cheap, high-precedent, config-only

**Q1. Per-layer SDClip + int7 emb on CaseOps stack** (re-port #1586)
- Source: PR-precedent (clean −0.011 on pre-CaseOps stack)
- Change: MLP k=12.0, attn k=13.0, emb k=15.0, EMBED_BITS=7 (currently 7 already? verify)
- Bytes: int7 emb saves ~530 KB → can fund higher LQER rank or other levers
- Δ guess: −0.001 to −0.003
- Risk: low (config-only)
- Cost: ~$10-15 (3-seed)
- **Caveat**: CaseOps changed embedding shape with control tokens — optimum may have shifted. Should sweep, not just re-use #1586's exact values.

**Q2. SpinQuant ON ablation in current baseline**
- Source: PR diagnostic — settles the rotation question in our exact stack
- Change: `SPINQUANT_ENABLED=1 SPINQUANT_SITES='attn_in,attn_proj_in,mlp_in,mlp_proj_in'`
- Bytes: 0
- Δ guess: ±0.001 (most likely null per prior evidence; cheap to confirm in current stack)
- Risk: low
- Cost: ~$10
- **Why bother**: prior nulls predate LQER + CaseOps + current Muon-WD-ratio. Worth confirming the fact applies post-LQER before truly closing the direction.

**Q5. Bigger calibration: 16 → 64 or 128 batches**
- Source: literature (Hubara et al. 2021, GPTQ paper used 128)
- Change: `GPTQ_CALIBRATION_BATCHES=128`
- Bytes: 0
- Δ guess: −0.001 to −0.003
- Risk: low (calibration is offline, doesn't blow training budget)
- Cost: ~$5
- **Cheapest experiment in the list.** GPTQ calib at 1/8 the literature default is plausibly leaving free signal on the table.

### Tier 2: novel, code-needed, high-upside

**Q3. Post-quant LN/bias fitting loop** (#1818-B design)
- Source: PR-novel — designed by taka6745, never implemented
- Change: After GPTQ baking, run 5 iterations re-fitting LayerNorm scales/shifts (and biases if present) against fp32 activations on calib data
- Bytes: 0 (LN params already exist)
- Δ guess: −0.005 to −0.020 (designer's projection — take with salt)
- Risk: medium (novel, designer's number unverified; but mechanism is principled)
- Cost: ~30 lines of code + $10 for screen
- **Highest-upside Tier-1/2 candidate.** No one has tried it. Even at the low end (−0.005) it'd be the biggest single quant lever since SDClip.

**Q6. OmniQuant** (learnable per-channel clipping + LET smoothing)
- Source: literature (Shao et al. 2024, ICLR)
- Change: replace GPTQ rounding with block-wise reconstruction; learnable per-channel weight clip thresholds + per-channel smoothing scale
- Bytes: ~0 (LET fuses into adjacent weights, LWC bakes into scales)
- Δ guess: −0.002 to −0.005
- Risk: medium (5-7d of code, untested in comp; OmniQuant on 50M scale unproven)
- Cost: ~$10-15 to validate
- **Defer until Tier 1 + Q3 land.**

### Tier 3: code-complete elsewhere, never run

**Q7. Group-size 128 GPTQ** (#1664)
- Source: PR-code-complete by zoharb157, never validated
- Change: GPTQ with group_size=128 instead of per-row
- Bytes: groupwise scales add ~2% storage overhead
- Δ guess: −0.002 to −0.005
- Risk: medium (interaction with SDClip unknown)
- Cost: ~$10

**Q8. Mixed int5 attn / int6 MLP / int7 emb** (#1817 pattern)
- Source: PR-precedent (works at 1h compute on different arch)
- Change: per-tensor-type bit allocation
- Bytes: byte-saving (int5 attn frees space for tighter MLP/emb)
- Δ guess: −0.002 to −0.005 if attention tolerates int5 here
- Risk: medium-high (#1646 showed wholesale int5 fails)
- Cost: ~$10
- **Caveat**: needs careful per-component damage measurement first; could regress badly.

**Q4. Short-burst AR self-gen calibration** (~30s reserve, not 210s)
- Source: PR + literature
- #756 showed AR matches val-calib within 0.0003 BPB
- #1234 failed only because 210s reservation cost 30% of training steps
- Change: ≤30s AR generation, smaller batch (16 batches × 2048 tokens)
- Bytes: 0
- Δ guess: −0.0003 to −0.001 (small but legal)
- Risk: low
- Cost: ~$5-7

**Q9. AdaRound** (Nagel 2020) layered on GPTQ output
- Source: literature, untried in comp
- Change: per-layer rounding-mask optimization with AdamW for 200-500 steps post-GPTQ
- Bytes: 0
- Δ guess: −0.001 to −0.004
- Risk: medium (may overlap heavily with GPTQ + LQER which already address these layers)
- Cost: ~$7
- **Defer — likely redundant with what LQER already corrects.**

### Tier 4: explicitly NOT to try

- **QuaRot / SpinQuant / DuQuant / Hadamard rotations** — null on Muon weights (4 ablations)
- **AQLM / QuIP# / QTIP / GPTVQ / SpQR** — codebooks add bytes, designed for int2-3 not int6
- **SmoothQuant / QDrop** — designed for activation quant, we're weight-only
- **AffineQuant** — subsumed by FlatQuant for our scale; affine bytes risk cap
- **BitNet** — requires retraining from scratch
- **TurboQuant** — already killed (PR #918, 41× worse quant penalty than int6)
- **CROWN-Q** — predates SDClip, never ported; small effect (~0.0005), unlikely to help
- **Naive AR self-gen with 210s reserve** (#1234 failure mode) — must be short-burst

## Suggested sequencing

1. **Q1 + Q2 + Q5 in parallel** (config-only, ~$30 total, ~1 wallclock day)
   - All three are cheap; results triangulate the "is there free signal in clip/emb-bits/calib?" question
2. **If any Tier-1 wins, fold into a new baseline before Tier-2**
3. **Q3 (post-quant LN fit)** — independent code work; spec separately. Highest expected upside.
4. **Q6/Q7/Q8** — only if Q1-Q3 don't get us close to #1801 (1.06287 quantized)

## Open questions to resolve before specing

- Is `EMBED_BITS=7` actually our current setting, or did I misremember? (verify in launch.out)
- Does CaseOps stack have biases on linears? (relevant for Q3)
- What's the actual current `GPTQ_CALIBRATION_BATCHES` setting? (16 in spec but verify)
- For Q3, what activations do we capture for the fit objective — last hidden state? per-block output?

## Reference checkpoint for quant-repair sweeps

**Use `045-loop-layer-improvements/armD`** as the iteration reference:

- Path on NE-1 volume: `/workspace/runs/045-loop-layer-improvements/armD/final_model.pt` (135 MB)
- Pre-quant EMA val_bpb: 1.06547 (+0.0003 vs canonical baseline — noise-level)
- Quantized val_bpb: 1.07467 (+0.0006 vs canonical)
- **Quant tax: +0.00920** (essentially identical to canonical baseline's +0.00896)
- Hardware: 4×H100, 5101 steps, 1199s wallclock
- Single lever active: `LOOP_SCALE_INIT=recip` (LOOP_ITER_EMBEDS=0, MLP_ONLY_FROM_PASS=0)
- Git commit: `1c6cd7c`

Why armD: canonical 039 baseline checkpoint was never saved (only train.log exists). armD is the closest full-run checkpoint we have. Quant-tax delta is what matters for repair experiments, and armD's quant-tax matches baseline within noise.

## Workflow

1. **Add `RESUME_FROM_CKPT` mode** to `train_gpt.py` (~20 lines): skip `train_model()`, load checkpoint into `base_model`, then proceed normally to pre-quant eval → `serialize()` (= GPTQ with current env vars) → deserialize + quantized eval. Cost: ~30 min code.
2. **Verify**: load armD's `final_model.pt`, run with current quant config, confirm reproduces armD's 1.07467 quantized number. If it does, infra works. ~$1.
3. **Sweep Tier 1** (Q5, Q1, Q2, Q4, Q5b, Q5c — all config-only): ~6 variants × ~$1 = ~$6 total, ~1h wallclock on a single 4×H100 pod.
4. **If any Tier 1 wins**, fold into a new combined config and re-test.
5. **Tier 2**: Q3 (post-quant LN/bias fit) is the dark horse — ~30 lines code + ~$1 to validate. Q6/Q7/Q9 only if Tier 1 + Q3 don't get us close to #1801 (1.06287 quantized).

## Cost comparison

| Mode | Cost per experiment | Wallclock |
|---|---|---|
| Full screen (current) | ~$5-10 | ~20 min train + ~1 min eval |
| RESUME_FROM_CKPT (proposed) | ~$1 | ~5-7 min (GPTQ + eval only) |

5-10× cost reduction. Lets us explore 20+ variants for the cost of 2 full screens.

## Bigger-picture caveat

Our quant tax is +0.009. Even a perfect quant repair (tax → 0) gets us pre-quant 1.06514 → post-quant 1.06514, which is **still 0.002 worse than #1801 (1.06287)** and 0.004 worse than #1797 (1.06157). Quant repair alone won't catch the frontier — it has to compose with pre-quant wins. Worth doing for the headroom it might free, not for being the single lever that wins.
