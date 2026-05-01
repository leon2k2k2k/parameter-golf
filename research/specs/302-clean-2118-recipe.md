# Spec 302 — Clean #2118 recipe (Gated XSA + token-only n-gram inside timer)

**Slug:** `302-clean-2118-recipe`
**Created:** 2026-05-02
**Status:** NEEDS RESTART — pilot seeds (42, 1234) non-submittable; see below
**Branch:** `main` (running directly from #2118 record dir in workspace)

---

## What this is

PR #2118's exact training code run on clean HF data (`romeerp/parameter-golf-caseops-v1`)
instead of the leaky local-prep dataset, with the config locked to match the submitted
#2118 recipe exactly: `compressor=pergroup` (lrzip), `SKYLIGHT_MUON=0`,
`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0` (inside timer).

## Pilot seeds — non-submittable

Seeds 42 and 1234 were launched with three incorrect settings vs submitted #2118:

| Setting | Pilot (wrong) | Target (correct) | Impact |
|---|---|---|---|
| `compressor` | `brotli` | `pergroup` (lrzip) | Artifact 16.95 MB > 16 MB cap — **fatal** |
| `skylight_muon_enabled` | `True` | `False` | Different training regime |
| `ngram_hint_precompute_outside` | `True` | `False` | Outside-timer precompute — not legal |

These seeds are finishing eval now. Their val_bpb is logged as a reference signal only.
**Do not submit.** Full restart required with corrected settings.

---

## Baseline

| PR | val_bpb | Data |
|---|---:|---|
| #2118 (submitted, LEAK) | 1.04350 | LEAK — ~0.015 bpb inflation from train/val overlap |
| #2014 (best clean) | 1.05759 | CLEAN HF |
| #1851/#1868 (merged SOTA) | 1.06128 | CLEAN HF |

Expected range on clean data: roughly #2014 minus Gated XSA contribution (~0.010), adjusted for
different hyperparams vs #2014. No firm prior — the run will tell us.

---

## Confirmed compliance (from live pod logs)

| Check | Status | Evidence |
|---|---|---|
| Data source | ✅ CLEAN | `datasets_dir: /dev/shm/pgolf_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` — HF download path |
| Train/val separation | ✅ CLEAN | `train_files: fineweb_train_*.bin`, `val_files: fineweb_val_*.bin` — disjoint files |
| val_tokens | ✅ CLEAN | `47,851,520` — canonical HF val, identical to #1851/#1868 |
| train_shards | ✅ | `80` — consistent with HF dataset |
| N-gram experts | ✅ TOKEN-ONLY | `within_boost: 0.0`, `word_boost: 0.0` — within-word and word-level experts disabled |
| Mid-run recompile | ✅ NONE | Zero recompile events in both logs |

---

## Compliance decisions (resolved)

| Issue | Decision |
|---|---|
| Skylight Muon | **OFF** — match submitted #2118 (`skylight_muon_enabled: False`) |
| N-gram precompute timing | **Inside timer** — `ngram_hint_precompute_outside: False`; required for legality |
| Compressor | **pergroup** (lrzip) — brotli blows past 16 MB cap |

---

## Full hyperparameter snapshot (from live pod logs, seed 42 and 1234 are identical)

```
# Model
num_layers: 11         model_dim: 512        num_heads: 8
num_kv_heads: 4        mlp_mult: 4.0         rope_dims: 16
loop_start: 3          loop_end: 5           num_loops: 2
parallel_start_layer: 8  parallel_final_lane: mean
enable_looping_at: 0.35

# Training
train_seq_len: 2048                  # fixed (no progressive schedule)
train_batch_tokens: 786432
max_wallclock_seconds: 600.0
gptq_reserve_seconds: 2.0
warmdown_frac: 0.75                  # vs 0.85 in #2014
beta2: 0.95                          # vs 0.99 in #2014
min_lr: 0.0                          # vs 0.1 in #2014
matrix_lr: 0.026
qk_gain_init: 5.0                    # vs 5.25 in #2014
grad_clip_norm: 0.3
matrix_clip_sigmas: 12.85
mlp_clip_sigmas: 10.0                # vs 11.5 in #2014
attn_clip_sigmas: 13.0
embed_clip_sigmas: 20.0

# Optimizer
skylight_muon_enabled: False         # match submitted #2118
muon_momentum: 0.97
muon_wd: 0.095

# Architecture features
gated_xsa_enabled: True              # per-head scalar; zero-init
smear_gate_enabled: True  gate_window: 12
skip_gates_enabled: True
sparse_attn_gate_enabled: True  sparse_attn_gate_scale: 1.0  # vs 0.5 in #2014
caseops_enabled: True
ema_decay: 0.9965
logit_softcap: 30.0

# Quantization / compression
matrix_bits: 6          embed_bits: 8         # vs int7 in #2014
lqer_enabled: True      lqer_rank: 4          lqer_asym_group: 64
lqer_top_k: 1           lqer_factor_bits: 4
awq_lite_enabled: True  awq_lite_group_size: 64  awq_lite_group_top_k: 1
compressor: pergroup                 # lrzip — required for artifact ≤16MB

# Eval / TTT
eval_seq_len: 2048                   # vs 2560 in submitted #2118, 3072 in #2014
eval_stride: 64
phased_ttt_num_phases: 1
phased_ttt_prefix_docs: 1000
ttt_lora_rank: 80
ttt_chunk_size: 48
ttt_batch_size: 64
ttt_k_lora: True  ttt_o_lora: True  ttt_mlp_lora: True
ttt_beta2: 0.999

# N-gram (token-only, outside timer)
ngram_tilt_enabled: True
ngram_hint_precompute_outside: False # inside timer — matches submitted #2118; required for legality
within_boost: 0.0                    # within-word expert DISABLED
word_boost: 0.0                      # word-level expert DISABLED
token_boost: 2.625
token_order: 16
token_threshold: 0.8
```

---

## Seed plan

| Seed | Status | Notes |
|---|---|---|
| 42 | ⚠️ PILOT — non-submittable | brotli + Skylight Muon + ngram outside timer |
| 1234 | ⚠️ PILOT — non-submittable | same |
| **42** | ⬜ RESTART needed | correct config (pergroup, no Skylight, ngram inside) |
| **1234** | ⬜ RESTART needed | correct config |
| **314** | ⬜ not started | third seed for 3-seed submission |

---

## Accept criteria

| Criterion | Threshold |
|---|---|
| val_bpb below clean baseline | < 1.057 (#2014) |
| val_bpb shows Gated XSA contribution | < 1.050 |
| val_bpb competitive with Spec 301 target | < 1.048 |
| eval_time (per seed) | ≤ 600s |
| artifact size (per seed) | ≤ 16,000,000 bytes |

---

## Monitoring

Seeds 42 and 1234 are at ~step 3000 (~5.7 min wallclock) as of spec creation.
Training ends at ~step 4900 (~9.9 min). Eval begins after GPTQ (2s reserve).
N-gram precompute (~130s) runs before eval clock starts.
Check logs for `ngram_tilt:precompute_done` and `ttt_phased:` lines at run end.

After both seeds finish: launch seed 314 on a new pod with identical env vars.
