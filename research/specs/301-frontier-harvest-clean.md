# Spec 301 — Frontier harvest: Gated XSA + progressive context on clean HF data

**Slug:** `301-frontier-harvest-clean`
**Created:** 2026-05-02
**Status:** DRAFT
**Branch:** `exp/301-frontier-harvest-clean` (to be created)
**Links:** `caseops-memory-leakage/verdicts.md`, Spec 300 (#2014 base)

---

## Hypothesis

Gated XSA (the dominant training lever in the leaky frontier, ~−0.010 BPB isolated from the
#2041 vs #2018 comparison) is a clean architectural addition. Stacking it on top of #2014's
already-clean progressive-context + short-doc TTT recipe should produce the best clean val_bpb
to date without touching n-gram (disabled entirely to avoid C1 legality and compile concerns).

**N-gram is OFF.** `NGRAM_TILT_ENABLED=0`. No n-gram code compiled or loaded.

---

## Baseline

| PR | val_bpb | Data | Notes |
|---|---:|---|---|
| #2014 (@simonbissonnette) | 1.05759 | CLEAN HF | Best verified clean; direct base |
| #1851/#1868 | 1.06128 | CLEAN HF | Merged SOTA |
| #2118-clean (in progress) | TBD | CLEAN HF | Spec A; full n-gram + Gated XSA |

---

## Levers

| Lever | Source | Δ BPB estimate | Notes |
|---|---|---|---|
| **Gated XSA** | #2018/#2118 (LEAK) | ~−0.010 | Isolated: #2041 (1.05692) vs #2018 (1.04722); same eval, same data, only diff is Gated XSA |
| Progressive 3k context | #2014 (CLEAN) | already in baseline | Inherited — not an incremental add |
| Short-doc TTT schedule | #2014 (CLEAN) | already in baseline | Inherited |

Expected val_bpb: **~1.045–1.050** (clean baseline 1.05759 minus ~0.010 Gated XSA).
Spec A (#2118-clean) result will sharpen this estimate once available.

---

## Config diff vs #2014

### Only addition

```bash
GATED_XSA_ENABLED=1    # per-head scalar gate on XSA; zero-init; pure superset of base
```

Everything else inherited from #2014 verbatim:

```bash
# Context schedule — unchanged
TRAIN_SEQ_SCHEDULE=1024@0.100,2048@0.700,3072@1.000
TRAIN_SEQ_LEN=3072
EVAL_SEQ_LEN=3072
TTT_EVAL_SEQ_LEN=3072
EVAL_STRIDE=1536
EVAL_INCLUDE_TAIL=1

# TTT — unchanged
PHASED_TTT_NUM_PHASES=1
PHASED_TTT_PREFIX_DOCS=2500
TTT_SHORT_SCORE_FIRST_ENABLED=1
TTT_SHORT_SCORE_FIRST_STEPS=256:8,2000:24
TTT_SHORT_DOC_LEN=2000
TTT_LORA_RANK=80
TTT_MASK=no_qv
TTT_Q_LORA=0
TTT_V_LORA=0
TTT_LOCAL_LR_MULT=0.75

# Optimizer / training — unchanged
WARMDOWN_FRAC=0.85
BETA2=0.99
MATRIX_LR=0.026
QK_GAIN_INIT=5.25
GRAD_CLIP_NORM=0.3
MIN_LR=0.1
GPTQ_RESERVE_SECONDS=4.0

# Quantization — unchanged
LQER_RANK=4
LQER_TOP_K=3
LQER_ASYM_GROUP=64
AWQ_LITE_ENABLED=1
EMBED_BITS=7

# Eval path — unchanged
ASYM_LOGIT_RESCALE=1
SMEAR_GATE_ENABLED=1
GATE_WINDOW=12
SKIP_GATES_ENABLED=1
SPARSE_ATTN_GATE_SCALE=0.5
EMA_DECAY=0.9965
FUSED_CE_ENABLED=1

# N-gram — explicitly disabled
NGRAM_TILT_ENABLED=0
```

---

## Compile safety

This is the main risk. Two compile concerns:

### 1. Progressive context + torch.compile shape changes

`TRAIN_SEQ_SCHEDULE` changes `seq_len` mid-run at `frac=0.10` and `frac=0.70`. Each new
seq_len triggers new compile shapes. **#2014 already solves this** via:
```
compile_shape_warmup: True
compile_shape_warmup_iters: 1
compile_shape_warmup_loop_modes: auto
```
This pre-warms all three seq_len shapes (1024, 2048, 3072) before training starts, so
no mid-run recompile occurs. **When porting from #2014's train_gpt.py, these warmup flags
must come along.** Verify in the mini log: no `Recompiling` lines after step 1.

### 2. Gated XSA — per-head scalar

`gated_xsa_enabled` adds a learnable `xsa_alpha` scalar per head per layer. Requirements:
- Must be registered as `nn.Parameter`, not computed inline
- Must follow the always-tensor pattern: inactive in non-XSA layers via buffer (not None)
- Zero-initialized (`tanh(0) = 0` → model starts identical to base at step 0)
- Must NOT be a weight slice inside a compiled region

Verify in mini: `gated_xsa_enabled: True` in hyperparams header; loss at step 1 matches
expected range (not NaN, not identical to non-gated run since gradients differ from step 1).

### 3. No n-gram C extension

With `NGRAM_TILT_ENABLED=0`, `online_ngram_state.c` must not be imported or compiled.
The import must be guarded by the flag, not unconditional. Verify: no `gcc` or `cffi`
output in the mini log.

---

## Code assembly

**Base:** #2014's `train_gpt.py` is the cleanest starting point (has progressive context +
short-doc TTT + compile warmup). Port Gated XSA from #2118:

1. Copy `gated_xsa_enabled` flag + `xsa_alpha` parameter registration from #2118's
   `train_gpt.py` into #2014's `train_gpt.py`.
2. Add the `tanh(xsa_alpha)` multiplication in the XSA subtraction path.
3. Guard the n-gram import in #2118 so it's skipped when `NGRAM_TILT_ENABLED=0` (or just
   don't port n-gram code at all — keep #2014's base clean).
4. Do NOT port any other #2118 additions (n-gram tilt, beta2 changes, etc.).

**Data:** same clean HF download as #2118-clean run:
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='/dev/shm/pgolf_data',
    max_workers=8,
)
"
```

---

## Hardware ladder

| Rung | Hardware | Wallclock | Purpose | Required? |
|---|---|---|---|---|
| **Mini** | 2×H100 | 120s | Compile warmup fires, Gated XSA trains, no mid-run recompile, no NaN | **Yes** |
| **Official** | 8×H100 | 600s | 3-seed submission | After mini passes |

---

## Seed plan

- Mini: seed 42, 120s wallclock
- Official: seeds 42, 314, 0 (match #2014 seeds for paired comparison)

---

## Mini checklist (before launching official)

- [ ] No `Recompiling` in log after step 1
- [ ] `gated_xsa_enabled: True` in hyperparams header
- [ ] `ngram_tilt_enabled: False` (or absent) in hyperparams header
- [ ] No gcc/cffi/C-extension compile output
- [ ] train_loss decreasing normally by step 20
- [ ] `compile_shape_warmup: True` in hyperparams header
- [ ] eval_time (extrapolated to 600s) < 600s

---

## Accept criteria

| Criterion | Threshold |
|---|---|
| val_bpb < clean baseline | < 1.057 |
| val_bpb shows Gated XSA contribution | < 1.052 |
| val_bpb competitive with leaky frontier (clean) | < 1.048 |

---

## Stop-early criteria

- NaN loss at any step → abort immediately
- Mid-run recompile detected → abort, fix compile warmup, relaunch
- val_bpb > 1.060 at end of mini → training signal broken

---

## Open questions for interview

1. Does #2014's train_gpt.py Gated XSA code exist at all, or is it a fresh port from #2118?
2. Is `compile_shape_warmup` already in the repo's baseline train_gpt.py or only in #2014's?
3. Should we match #2014's `GPTQ_RESERVE_SECONDS=4.0` or use #2118's `2.0`?
   (4.0 is safer; 2.0 gives 2 extra training seconds. Start with 4.0.)

---

## Cost estimate

| Rung | Time | Rate | Cost |
|---|---|---|---|
| Mini (2×H100) | ~15 min | ~$3/hr | ~$0.75 |
| Official (8×H100, 3 seeds) | ~3 × 25 min | ~$24/hr | ~$30 |
| **Total** | | | **~$31** |

---

## Follow-on

If Spec 301 lands < 1.048: next spec is MP3 marker-pair fusion (Spec 302) — vocabulary
surgery that compresses `[▁,TITLE]`/`[▁,ALLCAPS]`/`[▁,CAPNEXT]` bigrams into single
alias tokens, freeing 8.47% of train tokens for more unique docs per wallclock step.
Requires custom dataset prep but is byte-preserving (BPB computed on canonical UTF-8 bytes).
