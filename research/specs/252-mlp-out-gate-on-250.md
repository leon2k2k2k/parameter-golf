# Spec 252 — Per-block MLP output gate on spec 250 base

**Date:** 2026-05-01 (deadline window)
**Idea:** PR #1941 (MarioPaerle) — per-block MLP output gate, weight-learnable.
**Lineage:** Spec 250 (PR #1953 verbatim + LeakyReLU² slope 0.3, commit d102266)
+ a single Block-level architectural change. Stacks on spec 250's training arch.

## Hypothesis

PR #1941 added a tiny per-block per-token sigmoid gate on the MLP output —
13 params/layer (`CastedLinear(12,1, bias=True)`), reading the first 12
channels of the post-attn residual stream — and demonstrated a clean
**−0.00085 BPB matched-seed mean** vs PR #1886 (3 seeds, every seed beat
its match). The current V21+#1953 stack has gates on the attention path
(AttnOutGate, SmearGate, SparseAttnGate) but **no gate on the MLP output**.
This is the only obvious gate placement still untested. Adding it on top of
spec 250's slope-0.3 base should compose additively because the slope flip
is mid-MLP arithmetic and the gate is post-MLP scaling — independent axes.

Expected post-TTT 3-seed mean: **1.0571 ± 0.0007**.

## Baseline

**Spec 250 same-seed post-TTT BPB.** Per-seed paired Δ on identical
{42, 0, 1234} seeds — noise cancels.

Reference numbers (from spec 250):
- Spec 250 s42 post-TTT: **1.05797** (vs #1953 s42 1.05825 → −0.00028)
- Spec 250 3-seed mean (in-flight): predicted 1.0578 ± 0.0006
- PR #1953 3-seed: 1.05855 (std 0.00030)
- PR #1941 isolated lever delta: −0.00085 (matched 3-seed on #1886 base)

## Expected Δ vs spec 250 (same seed)

**−0.0003 to −0.0007 BPB**, medium confidence.

- For: PR #1941's −0.00085 isolated delta is well above noise; mechanism is
  novel-axis (MLP-side, not attn-side); 13-param init is do-no-harm
  (sigmoid(5)≈0.993, weight=0); composes orthogonally to slope-0.3 by
  construction.
- Against: PR #1941 was tested on the weaker #1886 base (1.0697); a
  stronger V21 base may absorb part of the gain. TTT's LoRA modules also
  see the gate now (via train/TTT path consistency) — may slightly shift
  TTT optimum, mostly negative interaction risk.

Author estimate of P(3-seed mean Δ ≤ −0.0003): **~55%**.
P(3-seed mean Δ ≥ 0): **~20%**.

## Accept criteria

- **Per-seed:** post-TTT BPB at gate-on ≥ −0.0003 below same-seed spec 250
  result (above noise threshold). Compute paired Δ per seed; report mean.
- **3-seed mean:**
  - mean Δ ≤ −0.0005 → **promote**: submit spec 252.
  - 0 ≥ mean Δ > −0.0005 → **wash**: submit spec 250 unchanged.
  - mean Δ > 0 → **kill**.
- All seeds clear 600s train cap, 600s eval cap, 16 MB artifact cap.
- Pre-quant BPB within 1σ (≈0.0006) of spec 250's pre-quant value at
  matched seed (sanity: training trajectory unchanged).
- Sanity: artifact size delta vs spec 250 < +1 KB (143 fp16 params via
  passthrough = ~286 bytes raw; lrzip should compress most away).

## Config diff (vs spec 250)

No env vars change. The gate is hardcoded ON in train_gpt.py — same
discipline as PR #1941, which left the gate always-on. Reverting the spec
= revert the commit.

## Code changes

- **Branch:** `exp/252-mlp-out-gate-on-250` (forked from spec 250 commit
  d102266).
- **Pinned commit:** `6ed741b351b8d8be8a8ff0dd799f6a6088147c5a` (pushed to `fork`).
- **Train script:** `records/track_10min_16mb/2026-04-30_spec250_LongCtx_NoQV_LeakyRelu03/train_gpt.py`
  (same path as spec 250).

Inline diff (Block class):

```python
# Block.__init__:
self.mlp = MLP(dim, mlp_mult)
+ self.mlp_gate_out = CastedLinear(12, 1, bias=True)
+ self.mlp_gate_out._zero_init = True
+ with torch.no_grad():
+     self.mlp_gate_out.bias.fill_(5.0)
self.attn_scale = nn.Parameter(...)

# Block.forward — replaces the one-line MLP residual with a 4-line gated version:
- x_out = x_out + self.mlp_scale[...] * self.mlp(self.mlp_norm(x_out) * ..., up_w, down_w)
+ mlp_out = self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
+ gate_in = x_out[..., :12].contiguous()
+ gate = torch.sigmoid(self.mlp_gate_out(gate_in))
+ mlp_out = mlp_out * gate
+ x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
```

The same gate insertion is replicated in 3 other forward paths to keep
train and TTT-eval semantics consistent (per the existing `sparse_attn_gate`
warning comment in code):

- `_parallel_block` (no-LoRA parallel path): gate_in = `mlp_read[..., :12]`
- `_block_with_lora` (TTT serial): gate_in = `x_out[..., :12]`
- `_parallel_block_with_lora` (TTT parallel): gate_in = `mlp_read[..., :12]`

## Hardware ladder

- **Mini (2×H100):** SKIP. Justification: 4-path consistent code change
  with do-no-harm init (sigmoid(5)≈0.993 ≈ identity at start, weight=0).
  Compile-graph footprint: one new `CastedLinear(12,1)` module per Block,
  one sigmoid, one elementwise mul — same kernel patterns as the existing
  AttnOutGate path. Triton autotune surface unchanged. Mini test would
  only verify "compiles and trains a few hundred steps" which the parse-
  check + structural-similarity argument already covers. **Confidence
  this skip is correct: high — but if seed 42 misbehaves at <step 1000
  (loss explosion, NaN, or step-time blow-up), abort and run mini.**
- **Official (8×H100):** 3 seeds {42, 0, 1234} matching spec 250's seeds
  for direct paired comparison.

## Seed plan

Three seeds, **42 first as smoke**. If seed 42 post-TTT BPB > 1.06000 (=
clearly regressing past noise), abort 0 and 1234. If 42 lands in
[1.05750, 1.05900], proceed to 0 and 1234 (sequential or parallel per
execution's call).

## Inputs (verbatim from spec 250 / PR #1953)

- `DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- **No hotstart** — 600s wallclock-cap from-scratch training.
- All env vars verbatim from spec 250 (PR #1953 reproduction settings):
  `CASEOPS_ENABLED=1`, `VOCAB_SIZE=8192`, `ITERATIONS=20000`,
  `MAX_WALLCLOCK_SECONDS=600`, `TTT_ENABLED=1`, `PHASED_TTT_ENABLED=1`,
  `PHASED_TTT_NUM_PHASES=3`, `PHASED_TTT_PREFIX_DOCS=2500`,
  `EVAL_SEQ_LEN=2560`, `TTT_EVAL_SEQ_LEN=2560`,
  `TTT_LOCAL_LR_MULT=0.75`, `QK_GAIN_INIT=5.25`, `TTT_NO_QV_MASK=1`,
  `TTT_V_LORA=0`, `MIN_LR=0.10`, `MATRIX_LR=0.026`,
  `MATRIX_CLIP_SIGMAS=12.85`, `EMBED_BITS=7`, `EMBED_CLIP_SIGMAS=15.0`,
  `GPTQ_RESERVE_SECONDS=8.0` (per #1953 reference).

## Checkpoints to emit

Same as spec 250: per-seed `final_model.pt` + per-step `submission.json` +
`train.log` written under `runs/252-mlp-out-gate-on-250/seed_<seed>/`.
Retain `final_model.pt` for 7 days post-eval (in case of follow-on TTT-
budget-arm experiments on the saved checkpoint).

## Stop-early criteria

- NaN in train_loss → abort.
- train_loss at step 500 > 1.5× spec 250 step-500 train_loss → abort
  (gate broke training trajectory).
- Step time at step 1000 > +5% vs spec 250 step-1000 step time → flag, do
  not abort (allowable compile-cache cost on first run; later seeds should
  be warm).
- Eval-time TTT compile burst >120s → flag but allow (first time on this
  arch).

## Cost estimate

- 8×H100 SXM single seed: ~$8 (600s train + 450–520s eval + ~5min
  setup/save).
- 3 seeds total: ~$24.
- Pre-warm of gate kernel: covered by training run (the gate fires every
  step from step 1).

## Extra artifacts

- `train_gpt.py` per-seed diff vs spec 250 (the +32/-6 commit).
- Pre-quant + post-quant + post-TTT BPB per seed (same as spec 250 schema).
- Artifact bytes per seed (sanity: gate adds ~143 fp16 params ≈ 286 raw
  bytes pre-compression).

## Open questions for interview

1. **Pod availability & timing** — spec 250 seeds 0/1234 may still be
  running on the same 8×H100. Should 252 wait for 250 to finish (cleaner
  pod allocation, ~30min latency) or run on a separate pod (parallel)?
2. **Gate channel choice (`:12`)** — verbatim from PR #1941, which itself
  copied the AttnOutGate's window. Locking it to 12 for the matched
  comparison; if seed 42 wins, **252-followup** could ablate {8, 16, 24}.
  Is that follow-on worth queuing now, or wait for 252 result?
3. **Failure-case** — if seed 42 lands in wash zone (Δ ∈ [-0.0003, 0]),
  do we still run 0 and 1234 for confirmation, or fall back to spec 250
  immediately and save $16?
