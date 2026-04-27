# Spec 051 — PPM-D: Per-Pass MLP Distinct (enc/dec bank split)

**Date:** 2026-04-28
**Branch:** `exp/051-perpass-mlp-untied` @ `6e6dd1e`
**Idea:** enc/dec MLP bank split on loop layers 3–5

## Hypothesis

Loop pass enc and dec visits currently share the same `mlp_up/down_bank[i]` weights — the MLP is fully tied across passes. By slicing each bank into two halves (first half for enc, second half for dec), enc and dec visits see genuinely different weights with zero parameter overhead. The model can specialize each half to its pass role without adding any parameters.

Motivation: 040C showed late-loop MLP (layer 5) can shrink to mlp_mult=3.4 with no quality loss, suggesting loop layers hold excess shared capacity that would be better split across passes. 050D confirmed that stacking extra parameters in the MLP forward path hurts throughput and doesn't transfer to val_bpb.

## Baseline

Spec 050A (`f5b8af8` baseline): pre-quant EMA val_bpb **1.06484**.

## Expected Δ

−0.001 to −0.003 pre-quant. Low confidence (novel axis, no prior comp), but zero parameter cost makes downside cheap.

## Accept criteria

- Pre-quant EMA val_bpb ≤ 1.064 (matches or beats 050A baseline).
- No compile hang at loop activation.
- Step time within 3% of 050A pre-loop throughput (PPM-D is gated off pre-loop, should be free).

## Config diff (vs spec 050A)

No env var changes. The split is always-on for loop layers when `looping_active=True`. Controlled entirely by code.

## Code changes

Branch: `exp/051-perpass-mlp-untied` @ `6e6dd1e`
Train script: `records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py`

Key changes vs `exp/050D-lora-untied` (which is the parent, retains inert LoRA path):

```python
# GPT.__init__ — store split params at init time:
self._loop_layer_start = h.loop_start
self._loop_layer_end = h.loop_end
self._loop_mlp_hidden_half = int(h.mlp_mult * h.model_dim) // 2

# Enc loop — first half of bank (enc specialization):
if self.looping_active and self._loop_layer_start <= i <= self._loop_layer_end:
    h2 = self._loop_mlp_hidden_half
    up_w = up_w[:h2, :]
    down_w = down_w[:, :h2].contiguous()

# Dec loop — second half of bank (dec specialization):
if self.looping_active and self._loop_layer_start <= i <= self._loop_layer_end:
    h2 = self._loop_mlp_hidden_half
    up_w = up_w[h2:, :]
    down_w = down_w[:, h2:].contiguous()
```

Parameter math: 2 × [h2, d] slices = original [2h2, d] — identical total params.
Contiguity: `up_w[:h2,:]` is contiguous (row slice); `down_w[:,:h2]` is a column slice → `.contiguous()` required for matmul.
Compile safety: gate on `self.looping_active` → pre-loop blocks pass full bank → single graph variant per loop state.

## Hardware ladder

- **4×H100 mini** — 20-min screen. Required (new code path).
- **8×H100 official** — if mini shows ≥ −0.001 improvement.

## Seed plan

Single seed (42) for mini.

## Inputs

`/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...` No hotstart.

## Checkpoints

`final_model.pt` from GPTQ.

## Stop-early criteria

- NaN or train_loss > 7.0 after step 200.
- Step time > 1.05× of 050A pre-loop baseline after loop activation (should be clean — same weight shapes, just sliced).
- Pre-quant EMA val_bpb > 1.068 at end of run.

## Cost estimate

~$3 (20-min screen, 4×H100).

## Open questions for interview

1. Pre-loop step time should be identical to 050A — confirm at step 100.
2. Post-loop step time: each MLP matmul is now half the hidden dim → kernel may be faster or hit a narrow-matmul efficiency cliff. Flag if step time changes more than 3% at loop activation.
3. Watch that `looping_active` gate fires correctly — verify enc path gets `up_w[:h2,:]` not full bank (can log `up_w.shape` at step 1 post-activation if unclear).
