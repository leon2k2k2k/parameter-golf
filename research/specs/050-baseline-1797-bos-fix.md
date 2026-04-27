# Spec 050 — Baseline migration to PR #1797 + BOS fix

**Date:** 2026-04-27
**Type:** Baseline migration (lands directly on `research` per CLAUDE.md exception).
**Idea source:** `pr-analysis/frontier-scan/2026-04-27.md` — actionable cluster.

## Hypothesis

Migrating our research training base from #1736 to **#1797 + SmearGate BOS-fix** (#1855's training-side lever) gives us a cleaner, lower-floor pre-training stack to layer subsequent specs (047 family, PPM-D port, etc.) on top of. We expect post-EMA pre-quant ≈ 1.067 (matching #1797 / #1787 ceiling) and post-TTT quantized ≈ 1.061 (#1797 minus the BOS-fix delta).

## Baseline (what's being replaced)

- `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/` (our #1736-derived stack)
- bpb: pre-quant ≈ 1.0654, quantized ≈ 1.0747, post-TTT ≈ 1.06549 (per #1736's 3-seed mean)

## Expected Δ

- post-TTT: −0.004 BPB (1.06549 → ~1.061)
- Confidence: medium-high. #1855 (3-seed mean 1.06108, std 0.00090) and #1851 (1-seed 1.06128) both land here on essentially this stack.

## Accept criteria

- 1-seed mini on 4×H100: post-EMA pre-quant within ±0.002 of 1.067, post-TTT within ±0.003 of 1.061. If both hit, promote to 3-seed.
- 3-seed mean on 8×H100: post-TTT mean ≤ 1.063 (i.e., beats #1736's 1.06549 by clear margin).
- Compile graph parity check: train.log shows no mid-run recompile (per memory: NEVER tolerate recompile).

## What's in the stack (vs #1736)

Inherited from #1797:
- **Polar Express NS coefficients** — 5 per-iter minimax NS tuples in Muon (vs single tuple × 5 iters)
- **MIN_LR=0.10** — warmdown LR floor (final 25% of training has useful gradient)
- **Sparse Attn Gate** — narrow gate `(8, 12)` instead of dense `(8, 512)`; ~44 KB artifact savings
- **Fused softcapped CE** — Triton kernel; training-only, eval path unchanged
- **Smear Gate** — content-conditioned gate over last 12 tokens of residual stream (orthogonal to AttnOutGate)
- **LQER asymmetric rank-4** — post-GPTQ low-rank recovery for int6 MLP rows; ~30 KB artifact, recovers ~0.009 BPB of int6 quant tax
- **TTT warm-start A** + **TTT_LORA_ALPHA=144** + **TTT_WEIGHT_DECAY=1.0** (from #1767, eval-only)

Added in this spec (from #1855):
- **SmearGate BOS-fix** — masks the previous-token term wherever the current position is BOS, applied symmetrically in `_forward_hidden` and `forward_ttt`. Fixes a real cross-document leak in packed val streams.

NOT inherited from #1855 (deferred for cleaner attribution):
- 9-hparam greedy bundle (WARMDOWN_FRAC=0.85, BETA2=0.99, MLP_CLIP=11.5, EMBED_CLIP=14.0, SPARSE_ATTN_GATE_SCALE=0.5, TTT_LORA_RANK=80, TTT_WD=0.5, TTT_BETA2=0.99, PHASED_TTT_PREFIX_DOCS=2500). Each can be re-introduced as a separate spec if needed.
- lrzip+ZPAQ per-group serializer.

## Code changes

- **Branch:** none — lands directly on `research` (baseline-migration exception per CLAUDE.md).
- **New directory:** `records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/`
- **Source:** PR #1797's `2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/` directory, copied verbatim.
- **Patch:** BOS-fix applied to `train_gpt.py` at two sites (line numbers approximate, post-copy):

```diff
-        # SmearGate (PR #1667). Inline gate compute with .contiguous() on the slice fed
-        # to the projection so torch.compile fullgraph is happy. lam=0 + W=0 -> identity
-        # at init. This block runs unconditionally on the smear path; the cat keeps
-        # position 0 untouched so causality holds.
+        # SmearGate (PR #1667) with BOS leak fix (PR #1855).
+        # Inline gate compute with .contiguous() on the slice fed to the projection so
+        # torch.compile fullgraph is happy. lam=0 + W=0 -> identity at init.
+        # The not_bos mask zeros the previous-token term wherever the current token is
+        # BOS — fixes cross-document leak in packed val streams.
         if self.smear_gate_enabled:
             sl = self.smear_lambda.to(dtype=x.dtype)
             gate_in = x[:, 1:, : self.smear_window].contiguous()
             g = sl * torch.sigmoid(self.smear_gate(gate_in))
-            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
+            not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
+            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1] * not_bos], dim=1)
```

Same patch in `forward_ttt`.

## Hardware ladder

- **2×H100 mini (proxy)**: skip — not feasible for full-model val_bpb (compile + dataset scale).
- **4×H100 mini**: 1-seed sanity at 600s wallclock. Expected post-TTT ~1.061 ± noise. **Required.**
- **8×H100 official**: 3-seed (314, 42, 1234) — same seeds as #1797. **Required for promotion.**

## Seed plan

- Mini: seed 42 only.
- Official: 314, 42, 1234 (matches #1797).

## Inputs

- Data: `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` (CaseOps lossless tokenized; same as #1797 / our existing baseline).
- Tokenizer: `records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- Hotstart: none. Train from scratch (10-min wallclock).

## Checkpoints to emit

- Final EMA model (post-warmdown): `final_model.pt` at submission dir
- Final quantized: `final_model.int6.ptz`
- No mid-training checkpoints.

## Stop-early criteria

- NaN in train_loss → halt
- val_bpb >1.5 at 50% wallclock → halt (compile/init bug)
- Step time >250 ms/step at 4×H100 → halt (compile pressure regression)

## Cost estimate

- 4×H100 mini: ~$3 (~10 min training + ~10 min eval + 5 min compile = ~25 min @ ~$8/hr)
- 8×H100 3-seed: ~$30 (~25 min × 3 seeds @ ~$24/hr)

## Compile / pre-warm

- **Required**: full-autotune prewarm of inductor cache for this commit (`TRITON_AUTOTUNE_NUM_RUNS=1` once, stash to volume per `tmp_exec/cache_stash.sh`).
- Cache key suffix: `_050_pr1797_bos`
- Per memory: NEVER tolerate mid-run recompile. Pre-warm both pre-loop and loop-active graphs before launch.

## Open questions for interview

1. **Compile cost validation** — Triton kernel + Smear Gate + BOS mask on top of our current compile baseline. Will full prewarm at 4×H100 hit ≥4.30M tok/s pre-loop (per memory ref)? Worth a tok/s sanity at step 100.
2. **Seed alignment** — should we match #1797's seeds (314, 42, 1234) or rotate to fresh ones (e.g., 0, 1, 2)? Matching makes Δ-vs-#1797 cleaner; fresh tests over-fit risk.
3. **Failure case** — if 4×H100 mini lands above 1.063 post-TTT (clear regression vs #1797's 1.06157), do we (a) halt, (b) verify the BOS-fix isn't the culprit by reverting to plain #1797, or (c) escalate to interview before re-running?
4. **Base-of-base** — once 050 lands, do all subsequent specs (047 family completion, PPM-D port, etc.) target 050 as their base? Or do we keep 047 work on the prior #1736 base for continuity?

## Next specs depending on this

- **051** — SmearGate BOS-fix isolated delta verification (optional sanity)
- **052** — PPM-D port from #1857 (contingent legal)
- **053** — `PHASED_TTT_PREFIX_DOCS=2500` (eval-budget converter from #1855)
- **054** — lrzip per-group serializer (from #1855)
