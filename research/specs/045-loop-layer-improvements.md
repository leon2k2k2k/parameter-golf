# Spec 045 — Loop Layer Improvements (A/B/C screen)

**Slug:** `045-loop-layer-improvements`
**Created:** 2026-04-26
**Status:** READY
**Branch:** `exp/045-loop-layer-improvements`
**Commit:** `1c6cd7cea0fa18ab07de120285c989df7c86b6bb`
**Links to:** `research/ideas/loop-layer-improvements.md`

## Hypothesis

Three independent levers targeting the Loop45 layers (4–5). Each default to disabled
and are byte-identical to baseline when off. Test each alone; best performers stack.

**A — Iteration embeddings:** Naive weight-tied loops can't distinguish pass 0 from pass 2.
ICML 2025 proves adding a timestep signal closes a fundamental approximation gap. A
zero-init `[num_passes, model_dim]` parameter injected at `loop_start` entry on each pass
costs ~1536 params and risks nothing.

**B — MLP-only loop:** Attention routes; MLP retrieves. After pass 0, re-running attention
may be redundant re-routing. Skipping attn on passes 2+ is cheaper per step → can afford
NL=3 or NL=4 for same throughput cost as NL=2 full-block. No prior art in ~300 PRs scanned.

**C — Residual 1/L init:** Weight-tied blocks repeated L times accumulate O(L) residual
norm at init. OpenReview 2026 prescribes 1/L init for `attn_scale`/`mlp_scale` in looped
layers. Pure init change, no new params, non-looped layers unchanged.

## Baseline

`runs/039-neg-slope-screen-on-1797-base/baseline/`
Pre-quant EMA val_bpb: **1.06514** | Quantized: **1.07410** | Steps: 5156

## Expected Δ

| Lever | Expected Δ | Confidence |
|---|---|---|
| A (iter embeds) | −0.001 to −0.003 | Medium (theoretically strong, empirically untested on our stack) |
| B (MLP-only) | −0.001 to −0.004 | Low-medium (novel, unknown risk) |
| C (1/L init) | −0.001 to −0.002 | Low (early-training stability, may wash out at 5k steps) |
| A+C stack | −0.002 to −0.004 | Medium |

## Accept criteria

- **Win:** pre-quant EMA ≤ 1.0641 (Δ ≤ −0.00104)
- **Noise:** 1.0641 – 1.0670
- **Kill:** ≥ 1.0670

## Config diff

All other config is identical to baseline (039b). Arms:

```
# Arm A: iteration embeddings
LOOP_ITER_EMBEDS=1

# Arm B: MLP-only loop (passes 2+)
MLP_ONLY_FROM_PASS=1

# Arm C: residual 1/L init
LOOP_SCALE_INIT=recip

# Arm A+C: natural stack
LOOP_ITER_EMBEDS=1  LOOP_SCALE_INIT=recip

# Arm B+C: MLP-only with stable init
MLP_ONLY_FROM_PASS=1  LOOP_SCALE_INIT=recip
```

Budget: run A and B first (highest signal). C is cheap to stack; run A+C or B+C if budget allows.

## Code changes

Branch: `exp/045-loop-layer-improvements`
Commit: `1c6cd7cea0fa18ab07de120285c989df7c86b6bb`

Key changes in `train_gpt.py`:

```python
# Hyperparameters additions (~line 361)
loop_iter_embeds = bool(int(os.environ.get("LOOP_ITER_EMBEDS", "0")))
mlp_only_from_pass = int(os.environ.get("MLP_ONLY_FROM_PASS", "0"))
loop_scale_init = os.environ.get("LOOP_SCALE_INIT", "ones")

# Block.forward: skip_attn flag (Lever B)
def forward(self, ..., skip_attn=False):
    if not skip_attn:
        attn_out = self.attn(...)
        x_out = x_in + self.attn_scale * attn_out
    else:
        x_out = x_in  # skip attention entirely
    x_out = x_out + self.mlp_scale * self.mlp(...)

# GPT.__init__: precomputed info lists (Levers A, B) + 1/L init trigger (Lever C)
# Forward: embed injection + skip_attn dispatch per step_idx
```

All three levers disabled by default → byte-identical to baseline on step 0.

## Hardware ladder

**Mini (required — code change):** 2×H100, 4h 20min wallclock, 1 seed per arm.
- Arms: A, B, A+C (3 arms minimum; add B+C or C-solo if time allows).
- Cost estimate: ~$7/arm × 3 arms ≈ **$21**. Run arms in parallel on 3 separate pods.

**Official:** 8×H100 full run, 3 seeds. Only for arms that clear the mini accept threshold.

## Seed plan

Mini: seed=42 for all arms (same as baseline). Official: seeds 42, 314, 1337.

## Inputs

- Data: `/workspace/parameter-golf/data/` (standard fineweb shards)
- Tokenizer: `/workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/tokenizer.model`
- No hotstart (full run from scratch, same as baseline)
- Config: same as 039b baseline, plus arm-specific env vars above

## Checkpoints to emit

Pre-quant EMA checkpoint + quantized blob (standard). No optim state needed on mini.

## Stop-early criteria

- NaN at any val eval → stop immediately
- val_bpb > 1.15 at step 1000 → stop (baseline ~1.24 at step 1000; 1.15 would be exceptional regression)
- Step time > 300ms sustained after loop activation → stop (throughput regression)

## Cost estimate

| Rung | Arms | Cost |
|---|---|---|
| Mini 2×H100 | 3 parallel arms (~21 min each) | ~$21 |
| Official 8×H100 | 1 arm × 3 seeds | ~$18 |
| Total (if one arm wins) | | ~$39 |

## Open questions for interview

1. Should arm B also test `MLP_ONLY_FROM_PASS=1, NUM_LOOPS=3` (extra MLP-only pass at low cost)? Budget allowing.
2. The `skip_attn` flag in Block.forward is a Python bool — does `torch.compile(fullgraph=True)` handle this cleanly, or does it cause unexpected recompilation? Execution should watch for extra compile events in the log.
3. Does the `loop_iter_embeds` parameter end up in the SCALAR or MATRIX optimizer group? It's shape `[num_passes, model_dim]` which is 2D — likely treated as a matrix by Muon. Verify in param-group debug logging.
