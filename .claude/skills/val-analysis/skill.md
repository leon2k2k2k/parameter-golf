---
name: val-analysis
description: Run a finished model on the validation set to produce per-token NLL data and a qualitative failure-analysis report. Optionally apply PPM-D byte-mixture (or other eval-time levers) and decompose the gain. Output goes to testing/outputs/<date>_<descr>/ with a markdown report + cached .npz of raw NLLs for fast re-analysis.
---

# Val-Analysis

A finished model gives us a `val_bpb` number. This skill turns that single number into **understanding** — *where* the model fails, *why*, and *how much downstream levers (PPM, TTT, etc.) help*.

## When to use this skill

- A new spec finishes and we want to understand its strengths/weaknesses qualitatively, not just the headline val_bpb.
- We're considering an eval-time lever (PPM, TTT tuning, score-first) and want to predict the gain on our model.
- We need to compare two checkpoints by failure profile, not just by aggregate score.
- A run produces a surprising val_bpb and we want to introspect what's happening.

Do NOT use this skill for:
- Running training (`tmp_exec/launch_*.sh` is for that).
- Producing official competition scores (use the eval pipeline in `train_gpt.py`).
- Quick smoke tests (use existing tools like `runs/<spec>/train.log`).

## Required inputs (interview the user)

Before running, ask:

1. **Which checkpoint?** Path to `final_model.pt`. Most checkpoints are at `/workspace/runs/<spec>/final_model.pt` on the volume. If user names a spec, find it via `find /workspace/runs -name final_model.pt`.

2. **Which `train_gpt.py`?** This is critical and easy to get wrong. The right one is **NOT** the worktree's top-level `train_gpt.py` — it's the file inside `records/track_10min_16mb/<base_dir>/train_gpt.py` that the launch script actually invoked. Confirm by reading `tmp_exec/launch_<spec>.sh` and finding the `TRAIN_SCRIPT="..."` line.

3. **PPM enabled?** If yes, we'll apply the PPM-D scorer (extracted from PR #1857) and report mix_bpb, gate_high_frac, ppm_only_bpb. If no, we report NN-only.

4. **What's interesting to inspect?** Default: top-50 worst predictions, per-category breakdown (URL/NUM/CODE/PROSE/HEX/PATH), best-25 contrast. Ask if user wants a specific slice (just URLs, just doc-boundaries, position trace through one doc).

5. **How many tokens?** Default: full val (40,540,160 tokens). Smaller (e.g. 200K) is ~10× faster but PPM tables don't fill up — only use for code testing.

## Required environment

- A running 1×H100 pod with the volume mounted (NE-1 region, parameter-golf template).
- If no pod exists, provision one: `runpodctl create pod --name pg-1h-inspect --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 1 --templateId y5cejece4j --imageName "runpod/parameter-golf:latest" --networkVolumeId hvpdph5i3g --dataCenterId US-NE-1 --secureCloud --startSSH --containerDiskSize 50 --volumePath /workspace`.
- Cost: ~$2.99/hr. A typical full-val run takes 6-7 min including model load + forward + PPM + report. Budget ≤ $0.50.

## Execution

**Use `testing/inspect_with_ppm.py`** as the canonical implementation. It already handles the methodology gotchas listed below. Don't rewrite from scratch.

```bash
# 1. Upload script + C source to pod
scp testing/inspect_with_ppm.py testing/ppm_scorer.c root@<pod-ip>:/tmp/

# 2. Run on pod
ssh root@<pod-ip> 'nohup python3 /tmp/inspect_with_ppm.py \
  --ckpt /workspace/runs/<SPEC>/final_model.pt \
  --train_gpt /workspace/<WORKTREE>/records/track_10min_16mb/<BASE_DIR>/train_gpt.py \
  --tokenizer /workspace/<WORKTREE>/<BASE_DIR>/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  --tokens 40540160 \
  --out /tmp/<SPEC>_full_val.md > /tmp/<SPEC>_run.log 2>&1 &'

# 3. Wait for completion (~6-7 min). Forward pass dominates.

# 4. Pull both files back
scp root@<pod-ip>:/tmp/<SPEC>_full_val.md  testing/outputs/<DATE>_<SPEC>_val_analysis/output.md
scp root@<pod-ip>:/tmp/<SPEC>_full_val.npz testing/outputs/<DATE>_<SPEC>_val_analysis/raw_nlls.npz

# 5. Commit + push (markdown only — .npz is too big for git)
```

The `.npz` is **the cache that matters** — once saved, all post-hoc analysis (different PPM hparams, different token-class breakdowns, different threshold sweeps) is seconds, not minutes. The forward pass itself is the expensive step.

## Methodology gotchas (lessons from 2026-04-28 first run)

These are the things that broke and ate hours of time. Internalize before running.

### 1. The right `train_gpt.py` is in the records/ subdir, NOT the worktree top-level

For example, 047B was launched from:
- `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`

Not `/workspace/pg-047B-loop-kv2-02977b7/train_gpt.py` (the top-level — that's a *different* incomplete code path lacking banks).

**Confirm via the launch script's `TRAIN_SCRIPT="..."` variable.**

### 2. Hparams must be seeded as env vars BEFORE importing the train_gpt module

The `Hyperparameters` class reads `os.environ` at import time. If env vars aren't set, you get the defaults (vocab=1024, num_layers=5) which won't match the saved checkpoint.

The script handles this via `seed_env_from_train_log()` which parses the saved `train.log`'s `Hyperparameters:` block and sets all values as env vars. Trust this — don't try to hand-construct the Hyperparameters object.

### 3. Two GPT class signatures exist

- Older record dirs: `class GPT: def __init__(self, h)` — pass the entire H object.
- Newer (047B-era worktree top-level): `class GPT: def __init__(self, vocab_size, num_layers, ...)` — pass kwargs.

The script tries `GPT(H)` first, falls back to kwargs.

### 4. `looping_active` must be set to True after construction

If `H.num_loops > 0` and `looping_active=False` (the default after construction), the model runs in non-looped mode — WAY higher val_bpb (e.g., 1.039 → 1.286 with banks-but-no-loop).

The script sets `model.looping_active = True` if `H.num_loops > 0`.

### 5. Val files have a 1024-byte header

`fineweb_val_*.bin` and `fineweb_val_bytes_*.bin` start with a 1024-byte header (256 int32s). Skip it before parsing as token data.

### 6. `val_bytes_*.bin` sidecar is uint16, NOT int32

This was the byte-counting bug. The sidecar holds per-instance bytes as uint16 (max 17 per token in our val). Reading it as int32 gives garbage values 4 orders of magnitude too large.

For CaseOps tokenized val, the sidecar is **the authoritative source** of bytes-per-token. `len(piece.encode("utf-8"))` undercounts by ~8% because CaseOps's capital markers attribute bytes per-instance, not per-token-id.

### 7. Use `_build_cu_seqlens` from the train_gpt module for varlen attention

The official `eval_val` builds a cu_seqlens mask based on BOS positions in each chunk and passes it to `forward_logits`. This makes attention RESET at document boundaries — without it, the model gets fake context from previous unrelated docs, inflating the NN's apparent quality.

The script imports `mod._build_cu_seqlens` directly. If absent, fall back to no-mask attention but flag a warning — numbers won't match official.

### 8. Save raw NLL data IMMEDIATELY after forward + PPM

A crash in the report-writing section can lose 5-7 min of compute. The script writes `.npz` *before* building the markdown lines. Match this pattern in any forks.

### 9. Numerical drift vs official ~0.012 BPB

Even after fixes 1-8, our reported val_bpb will sit ~0.012 BPB above the official value due to:
- Eager forward vs `torch.compile`d
- Explicit `.bfloat16()` vs `torch.autocast(bf16)`
- Single-seq chunks vs batched 32-seq chunks

This is acceptable for failure analysis. If exact reproduction is required, mirror the official `eval_val` closely (compile + autocast + batched chunks).

## Output format

Markdown report with these sections:

```markdown
# PPM-D byte-mixture inspection: <SPEC>

**Checkpoint:** <path>
**Tokens scored:** <N> (bytes: <M>)
**PPM config:** order=4 λ_hi=0.9 λ_lo=0.05 threshold=0.9

## Headline
| Metric | bits/byte |
|---|---:|
| NN only (`nn_byte_bpb`) | <X> |
| PPM only (`ppm_only`) | <Y> |
| NN + PPM mix (`mix_bpb`) | <Z> |
| Δ from PPM | <Z - X> |
| Gate high-confidence fraction | <%> |

## Comparison to dexhunter's #1857
[Side-by-side table for sanity-check]

## Per-category contribution to NN-only val_bpb
| category | count | % positions | mean bits/byte | contribution |
[URL, NUMERIC, CODE, HEX, PATH, PROSE rows]

## NLL distribution (NN only, bits/byte)
[Histogram buckets: 0.0-0.5, 0.5-1.5, 1.5-3.0, 3.0-5.0, 5.0+]

## Top-50 most catastrophic NN predictions
[Table: pos, NLL, actual, top-1 prediction, category, left context]

## Worst-10 per category
[Same as above, grouped by URL/NUMERIC/CODE/etc.]

## Top-25 best NN predictions (contrast)
[Same table format]
```

The `.npz` cache contains: `tokens`, `nll_nats`, `top_ids` (top-5), `top_logp` (top-5), `val_bytes_per_tok`, `ppm_out`, `ckpt_path`, `ppm_config`. ~1 GB compressed for full val.

## Comparing two models

If the user wants to compare model A vs model B (e.g., 047B vs 050 once 050 finishes):

1. Run val-analysis on both, getting `output_A.md` + `raw_A.npz`, `output_B.md` + `raw_B.npz`.
2. Compute NLL diff per token: `nll_A - nll_B`. Show:
   - Where model B improves most (largest negative deltas)
   - Where model B regresses (positive deltas) — surprising failures
   - Per-category where deltas concentrate
3. Write a comparative report to `testing/outputs/<DATE>_<A>_vs_<B>/comparison.md`.

This tells us whether 050 (say) is better at semantics, surface, doc boundaries, or just uniformly slightly better.

## What good looks like

- Headline numbers reported. PPM gain quantified if applicable.
- Per-category breakdown shows where val_bpb actually comes from (typically PROSE = ~85%, surface categories = ~15%).
- Top-50 catastrophic table is interpretable — surprises should be plausible (URLs, numerics, doc boundaries).
- `.npz` saved. Future PPM hparam sweeps don't need to re-run the model.
- Markdown report is committed to `research`. `.npz` is local-only (gitignored or kept in `testing/outputs/` outside git).

## What NOT to do

- Don't try to reconstruct Hyperparameters by hand — use env-seeding from train.log.
- Don't skip the sidecar bytes for byte counting — `len(piece.encode())` is wrong for CaseOps.
- Don't run without setting `looping_active=True` — you'll get garbage val_bpb.
- Don't write the markdown before saving the raw `.npz` — crashes lose hours of compute.
- Don't compare absolute val_bpb to official numbers without acknowledging the ~0.012 BPB methodology drift.
