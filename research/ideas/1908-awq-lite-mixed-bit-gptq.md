# Idea — Port PR #1908: activation-aware mixed-bit GPTQ on top of #1855 base

**Date:** 2026-04-29
**Source:** PR #1908 (romeerp), opened 2026-04-28, OPEN as of this writing.
**Provenance:** Author built directly on PR #1855 (our exact research baseline) and reports a **3-seed step-matched delta of −0.000265 BPB post-TTT** (1.06108 → 1.06081), with a larger −0.000196 BPB recovery on quantization tax alone (visible upstream of TTT).

## Why this lever

Of the levers that have appeared on the leaderboard since #1855, this is the only one that:

1. **Was step-matched on our actual base.** Author re-ran #1855's three published seeds at their exact stop steps (4945 / 4932 / 4917) and compared quantized + post-TTT bpb at those points. No "different fork, hope it transfers" risk.
2. **Has 3-seed validation already.** std=0.00089, mean delta −0.000265 → ~3σ below #1855. Clears the noise floor we've been stuck under with single-seed 060B/D/E/F runs (each ~−0.0002, ~1.5σ).
3. **Is purely quant-side.** Doesn't touch training, doesn't change the .pt checkpoint shape. Compatible with `RESUME_FROM_CKPT` from any 060A run.
4. **Doesn't overlap any lever we've already tried.** Orthogonal to SDClip (060B/D/E/F), deploy-time-repair (060C, didn't work), Partial SpinQuant (060G, refuted), EMBED_BITS=6 (060H, deprecated).

## Mechanism (in 5 bullets)

1. During the existing GPTQ calibration pass, also collect per-input-channel activation RMS (`act_rms`) for every quantized matmul.
2. For each layer, scan its weight columns in groups of 64. Score each group by `saliency = act_rms * mean(|w|)`, summed within the group.
3. Pick the **top-1** group/layer (`AWQ_LITE_GROUP_TOP_K=1`).
4. Inside the *same* full-tensor GPTQ Hessian solve, quantize that one group at **int8** (`AWQ_LITE_BITS=8`) instead of int6. The rest of the layer stays int6.
5. Stock #1855 LQER (rank=4, top_k=3, asym, group=64, factor_bits=4) kept on top of this AWQ-aware base.

The intuition: AWQ-style saliency identifies the column groups whose quantization errors get amplified most by activations. Spending +2 bits on the worst-offender group per layer recovers more error than uniformly spending the same bytes elsewhere.

## Cost vs gain

- **Bytes:** +~16 KB per seed (artifact mean 15,901 KB → 15,989 KB on author's runs). Under cap on all 3 seeds.
- **Code change:** ~100 lines:
  - `_awq_lite_group_candidates(w, act_rms, group_size)` helper (~15 lines)
  - Activation-stat collection inside GPTQ calibration loop (~10 lines)
  - Mixed-bit support in the Hessian-solve quant path (~30-50 lines)
  - 4 env var hyperparameters
  - `FORCE_STOP_STEP` for matched-step eval (~5 lines, optional)
- **Compute:** zero added training cost (weights unchanged). GPTQ pass slightly slower from extra act-stat collection. Eval-only via RESUME_FROM_CKPT possible — ~$1-3 per arm.
- **Reported gain:** −0.000265 BPB post-TTT (3σ on 3-seed std=0.00089).

## Stackability

- ✓ **With clip tightening (060B/D/E/F):** orthogonal — clip changes affect int6 quantization parameters; AWQ-lite changes which columns are int8. Both make GPTQ output less lossy through different mechanisms. Predict additive.
- ✓ **With LQER bumps (060F):** orthogonal — LQER corrects post-quant error globally; AWQ-lite localizes pre-quant precision. Predict additive.
- ✗ **With deploy-time AR-self-gen quant repair (046L / 060C):** unknown; both modify GPTQ calibration data. Don't stack until tested in isolation. (Moot — 060C didn't work for us anyway.)

## Open questions

1. **Does #1855's `pergroup` lrzip compressor give the same +16 KB cost on our hardware?** Author measured on their hardware; could be higher on ours if entropy of int8 vs int6 differs. Smoke first.
2. **Does the saliency score behave well on layers with our gating** (sparse_attn_gate, smear_gate)? Author's #1855 base has these; should work, but verify via post-quant val_bpb sanity.
3. **TOP_K=1 vs higher**: author's ablation table only shows TOP_K=1. Higher K is more bytes for more saliency coverage — worth a phase-2 sweep if TOP_K=1 lands.

## Risks

- **Code complexity in the GPTQ solve.** Mixed-bit-per-group inside a single Hessian solve is the trickiest part. If we get it wrong, post-quant val_bpb explodes. Smoke against author's reported per-seed quantized_bpb numbers (1.07226 / 1.07404 / 1.07427 at FORCE_STOP_STEP) before trusting it.
- **Pulling in author-private dataset path.** Their `requirements.txt` references `romeerp/parameter-golf-caseops-v1` on HF. We use the local NA-1 path; should be a no-op as long as we don't run from their record dir verbatim.

## Plan: spec as 060I

- 1-arm spec, eval-only via RESUME_FROM_CKPT against an existing 060A `final_model.pt`.
- Smoke: 1 seed, full TTT eval, verify post-TTT val_bpb lands within ~0.0003 of author's reported per-seed numbers.
- If smoke passes → 3-seed run for defensible delta vs #1855.
- If 3-seed lands ≤ 1.06081 ± 0.001 → stack with 060B (ATTN clip 12.5) as 060K.
