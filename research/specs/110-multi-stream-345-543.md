# Spec 110 — Multi-stream loop body (345 + 543 dual paths, MLP merge)

**Status:** FROZEN — code committed at `a0c1ba9` on `exp/110-multi-stream-345-543`,
pushed to fork. Ready to run.

**Date:** 2026-04-30 (architectural research thread, post-overnight wakes)
**Branch:** `exp/110-multi-stream-345-543`
**Pinned commit:** `a0c1ba94a1e29a8a8fb20fe0a7f56ae2c0745174`
**Parent:** 060A (#1855 port; forks from `exp/060-resume-ckpt @ a0a48b7`).

## Hypothesis

Genuine architectural change to the recurrence structure. Where the
canonical loop runs **3 sequential passes** through layers {3,4,5} in
order:

```
x_in → [3,4,5] → x_1 → [3,4,5] → x_2 → [3,4,5] → x_out
```

(9 layer-applications, all serially dependent), this spec runs **two
independent forward paths** through the same loop band in *opposite
orderings*, then merges them:

```
                ┌── [3 → 4 → 5] (stream A, canonical order) ──┐
x_in → split ──┤                                              ├── MLP merge → x_out
                └── [5 → 4 → 3] (stream B, reverse order) ────┘
```

(6 layer-applications total, two independent paths.)

**Both streams use the same trained bank weights** (`qo_bank`,
`kv_bank`, `mlp_up_bank`, `mlp_down_bank` rows for layers 3, 4, 5).
What differs is the iteration order. Composition `f₃ ∘ f₄ ∘ f₅` ≠
`f₅ ∘ f₄ ∘ f₃` in a transformer (non-commutative due to residual
updates and non-linearities), so the streams compute genuinely
different functions of the same input.

The bottleneck-MLP merge then learns a per-token, per-channel gate
to combine the two streams.

### What this is testing

1. **Architectural:** does forcing layer-order invariance during
   training regularize the model's loop weights into more general
   processors?
2. **Empirical:** is two parallel shallow refinements (NL=1 each)
   net-equivalent or better than one deep refinement (NL=2)?

This is a *training-from-scratch* spec — both streams contribute to
the loss during training, so the model's weights learn to produce
useful output regardless of which order they're applied. Cannot be
tested usefully eval-only on 060A (trained with order 3→4→5; reverse
order would be OOD).

## Baseline

060A trained from scratch (single seed). Pre-quant post-EMA val_bpb
~1.06358.

## Expected Δ vs 060A

| Outcome | Δ pre-quant val_bpb | Likelihood |
|---|---|---|
| Best (order-invariance regularizes) | -0.0010 to -0.0030 | low |
| Plausible (two NL=1 ≈ one NL=2 in expectation) | -0.0005 to +0.0005 | medium |
| Likely (sequential depth genuinely matters) | +0.0005 to +0.0030 | medium-high |
| Worst (reverse-order forces underfitting) | +0.0030 to +0.0100 | medium |

## Accept criteria

- **Win:** post-quant + post-TTT val_bpb ≤ (060A canonical) − 0.0005
- **Noise:** ±0.0005
- **Kill:** ≥ canonical + 0.0010

## Architectural design

### Stream A — canonical order (3→4→5)
Same as canonical NL=1: one forward pass through layers 3, 4, 5 using
their trained banks.

### Stream B — reverse order (5→4→3)
One forward pass through layers 5, 4, 3 using the *same trained banks*.
Layer 5's bank applied first (operating on raw `x_in`), then layer 4's,
then layer 3's.

### Merge — bottleneck MLP (option 3)

After both streams complete:

```python
delta = x_A - x_B                               # what differs
hidden = F.relu(self.merge_w1(delta))           # Linear(d=512 → r=8)
gate = torch.sigmoid(self.merge_w2(hidden))     # Linear(r=8 → d=512)
x_out = gate * x_A + (1 - gate) * x_B
```

- `merge_w1`: `nn.Linear(d, r)` — 512 × 8 = 4,096 params
- `merge_w2`: `nn.Linear(r, d)` — 8 × 512 = 4,096 params
- Total new params per loop layer: ~8K
- Across 3 loop layers (one merge module per loop layer): ~24K params total

### Init

`merge_w2` weights init to **zero**. Then sigmoid(0) = 0.5 → gate = 0.5
at init → `x_out = (x_A + x_B) / 2` (plain averaging at init, no
training signal to merge MLP yet).

### Stream tying

- Tie 1 (weight sharing): YES — both streams use the same bank rows.
- Tie 2 (mid-stream interaction): NO — streams run independently from
  `x_in` to merge.

## Config diff vs 060A canonical

```
MULTI_STREAM_LOOP_ENABLED = 1
LOOP_START = 3, LOOP_END = 5     (canonical, defines which layers are dual-streamed)
NUM_LOOPS = 0                     (no further outer iteration; multi-stream replaces NL=1 with dual NL=1)
ENABLE_LOOPING_AT = 0.35          (canonical activation point)
```

Other configs from 060A unchanged: SP8192, model_dim=512, attn config, etc.

## Code changes

Branch: `exp/110-multi-stream-345-543` (forks from 060A code @ a0a48b7).

### Files modified

- `records/track_10min_16mb/2026-04-29_PR1855_Port_Baseline/train_gpt.py`:
  - New env var `MULTI_STREAM_LOOP_ENABLED` (default 0)
  - New module `MultiStreamMerge(d, r)` — implements the bottleneck-MLP merge.
  - In `GPT.__init__`: when enabled, instantiate one `MultiStreamMerge` per loop layer (3 instances).
  - In `_forward_hidden`: when active and `looping_active`, replace the canonical loop iteration over `encoder_indices` with a dual-stream block (described below).
  - Same for `forward_ttt`.

### Estimated LOC

- env var + flag: ~5
- `MultiStreamMerge` module: ~25
- `_forward_hidden` dual-stream branch: ~50
- `forward_ttt` mirror: ~30
- Total: ~110 LOC

### Compile-graph audit

- `MULTI_STREAM_LOOP_ENABLED` is an `__init__`-time flag; it determines whether the merge modules are instantiated and used. Different value = different model architecture (different graph), but for a single run, the value is fixed at module construction.
- The dual-stream forward iterates layers 3, 4, 5 in order [3,4,5] for stream A and [5,4,3] for stream B — both are Python list iterations at trace time. Each produces its own block.forward calls; compile sees a single graph variant per (multi_stream_enabled state).
- Bank weight reads are unchanged (same `qo_bank[layer_idx]`, etc.). No weight slicing.
- Merge: tensor shapes are fixed (B, T, d=512). Linear layers have fixed param shapes. No dynamic shapes.
- Init: `merge_w2.weight = 0` zero-init means at first forward, gate = 0.5 → averaging. Behavior at init is mathematically a deterministic function of the streams — single graph variant.
- TTT path: same dual-stream structure under `forward_ttt`. LoRA application unchanged (per-block LoRA still applies to each block.forward call regardless of order). Compile burst at first eval, single variant cached.

**No mid-run recompile possible.** Audit passes the file's hard rules.

### Always-tensor compliance

- No None passthroughs of loop-only kwargs.
- Stream output tensors always exist (computed both streams every forward).
- Merge gate is a learned function of the difference; deterministic compute path.

## Hardware ladder

- **Mini rung: 4×H100, 5-min smoke** — required (significant code change).
  Verify: compile success, no NaN on first batch, gradient flows to merge MLP, no recompile.
- **Official rung: 8×H100, 1 seed (42), 600s train + ~400s eval.**

## Seed plan

1 seed (42) for screening. Multi-seed if the screen wins.

## Inputs

Standard 060A paths. No hotstart (fresh training run).

## Stop-early criteria

- train_loss > 5.0 at step 1000 → kill
- pre-quant EMA val_bpb > 1.080 at step 5000 → kill
- mid-run torch.compile recompile → kill (always-tensor violated)
- Compile fail on smoke → kill, debug

## Cost estimate

~$1 mini + ~$5 official = **~$6 single-seed.**

## Followup specs

- If 110 wins: 110-promo at multi-seed. Spec 112 candidate = vary the merge MLP rank (try r=16).
- If 110 ≈ canonical: layer-order invariance is a regularizer of approximately neutral value; close that thread.
- If 110 loses: deep sequential dependency in the recurrence is load-bearing; reverse-order doesn't compose.

## See also

- `research/ideas/parallelize-deep-looks.md` — parent ideas thread
- `research/specs/111-anderson-acceleration.md` — sibling architectural spec
  (different mechanism: math change to recurrence iteration rule)
