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

Seeds 42 and 1234 were launched with six incorrect settings vs submitted #2118:

| Setting | Pilot (wrong) | Target (correct) | Severity |
|---|---|---|---|
| `compressor` | `brotli` | `pergroup` (lrzip) | **Fatal** — artifact 16.95 MB > 16 MB cap |
| `ngram_hint_precompute_outside` | `True` | `False` | **High** — outside-timer precompute not legal |
| `min_lr` | `0.0` | `0.1` | **High** — LR floor matters a lot for final steps |
| `skylight_muon_enabled` | `True` | `False` | **High** — different training regime |
| `eval_seq_len` | `2048` | `2560` | **Medium** — shorter context costs val_bpb |
| `embed_bits` | `8` | `7` | **Medium** — int8 embed larger; pergroup+int7 gets under 16 MB |

These seeds are finishing eval now. Their val_bpb is logged as a reference signal only.
**Do not submit.** Full restart required with all six settings corrected.

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

## Restart config — fixes vs pilot

N-gram is turned **entirely off** (`NGRAM_TILT_ENABLED=0`, the default). This eliminates the
precompute timing concern, the within/word C1 debate, and the `WITHIN_BOOST`/`WORD_BOOST` env
vars — one flag kills all of it. The ~130s of eval budget recovered goes to TTT instead.
Trade-off: ~+0.005 bpb vs a working token-order-only tilt, accepted for cleanness.

```bash
NGRAM_TILT_ENABLED=0        # kill entire n-gram path; no precompute, no timing concern
COMPRESSOR=pergroup         # was brotli — lrzip required for artifact ≤16 MB
MIN_LR=0.1                  # was 0.0 — LR floor critical for final steps
SKYLIGHT_MUON=0             # was 1 — match submitted #2118
EVAL_SEQ_LEN=2560           # was 2048 — match submitted #2118
TTT_EVAL_SEQ_LEN=2560       # was 2048
EMBED_BITS=7                # was 8 — int7+pergroup keeps artifact under 16 MB
```

Drop from env (moot with NGRAM_TILT_ENABLED=0):
`NGRAM_HINT_PRECOMPUTE_OUTSIDE`, `WITHIN_BOOST`, `WORD_BOOST`, `TOKEN_BOOST`, `TOKEN_ORDER`, `TOKEN_THRESHOLD`

Everything else from the pilot was already correct:
clean HF data, `gated_xsa_enabled=True`, `gptq_reserve_seconds=2.0`, `lqer_top_k=1`,
`phased_ttt_prefix_docs=1000`, `phased_ttt_num_phases=1`.

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
min_lr: 0.1                          # match submitted #2118
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
matrix_bits: 6          embed_bits: 7         # match submitted #2118; int7+pergroup keeps artifact ≤16 MB
lqer_enabled: True      lqer_rank: 4          lqer_asym_group: 64
lqer_top_k: 1           lqer_factor_bits: 4
awq_lite_enabled: True  awq_lite_group_size: 64  awq_lite_group_top_k: 1
compressor: pergroup                 # lrzip — required for artifact ≤16MB

# Eval / TTT
eval_seq_len: 2560                   # match submitted #2118
ttt_eval_seq_len: 2560
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
