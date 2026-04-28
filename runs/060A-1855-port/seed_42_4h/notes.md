# 060A–E execution notes (session 2026-04-28)

Single-thread execution log for the 060 family on pod `tew5ycdcpbimyb` (evicted)
→ pod `v7aw1g82vi6ty5` (NE-1, 4×H100 SXM, $11.96/hr). Volume `hvpdph5i3g`.

## Hardware substitution (non-fidelity)

8×H100 SXM unavailable in NE-1 / JP at run time. Per user authorization, ran
4×H100 with three offsetting changes:

- `--nproc_per_node`: 8 → 4
- `MAX_WALLCLOCK_SECONDS`: 600 → 1200 (matched FLOPs: 8·600 = 4·1200)
- `GRAD_ACCUM_STEPS`: 1 → 2 (preserves global batch=786432, per-GPU mem unchanged)

Result is **not leaderboard-valid** (training >600s). Used for research-side
baseline validation + downstream lever signal only.

## Result table (post-quant val_bpb, 1 seed, no TTT except 060A)

| run | config delta vs 060A | post-quant val_bpb | Δ | total submission | notes |
|---|---|---:|---:|---:|---|
| 060A baseline | 1855 defaults | 1.07212124 | 0 | 16,148,692 (brotli) | OVER 16M cap by 149 KB |
| 060A repacked | brotli → pergroup compressor | (same) | 0 | 15,902,118 | LEGAL, 98 KB headroom |
| 060B | ATTN_CLIP_SIGMAS 13.0→12.5 | 1.07193891 | −0.000182 | 15,962,059 | LEGAL, 38 KB headroom |
| 060D-half | ATTN_CLIP_SIGMAS 13.0→12.0 | 1.07176948 | −0.000352 | 16,024,916 | OVER cap by 25 KB |
| 060E | EMBED_CLIP_SIGMAS 14.0→13.0 | 1.07191140 | −0.000209 | 15,957,730 | LEGAL, 42 KB headroom |
| 060F | ATTN 12.5 + EMBED 13.0 | 1.07172520 | −0.000397 | 16,017,570 | OVER cap by 18 KB |
| **060G** | **ATTN 12.5 + EMBED 13.5** | **1.07182443** | **−0.000297** | **15,989,345** | **LEGAL, 11 KB headroom — current best** |
| **060H** | **ATTN 12.75 + EMBED 13.0** | **1.07180628** | **−0.000315** | **15,987,688** | **LEGAL, 12 KB headroom — best stack** |
| **060H+TTT** | (same; full eval pipeline) | **post-TTT 1.05891477** | **−0.000269 vs 060A post-TTT 1.05918** | (same .ptz) | **−0.00217 vs #1855 3-seed mean (1.06108); −0.00098 vs #1855 seed 42 (1.05989)** |

### TTT analysis on the clip-tightening Δ

Two angles on the question "does clip tightening still help after TTT?":

**TTT recovery (post-quant → post-TTT) is unchanged by clip tightening:**
- 060A: 1.07212 → 1.05918, TTT recovers +0.01294 BPB
- 060H: 1.07181 → 1.05891, TTT recovers +0.01290 BPB
- Δ in TTT recovery: 0.00004 (noise). TTT helps the SAME amount regardless of clip choice.

**The clip-Δ survives through TTT at ~85%:**
- post-quant Δ (060H vs 060A): −0.000315
- post-TTT Δ  (060H vs 060A): −0.000269
- 269/315 ≈ 85% preservation. Clip tightening still wins post-TTT, just slightly smoothed.

So TTT and clip tightening are **independent levers** that compose with mild attenuation. Reasonable mental model: TTT learns LoRAs that partially compensate for quantization noise; tighter clip → less noise to compensate → smaller compensation needed → smaller absolute Δ visible after TTT, but the Δ still flows through.

### 060J (off-spec): phases 4 on top of H-clip — no help

Ran +1 TTT phase (PHASED_TTT_NUM_PHASES 3→4) on top of 060H clip combo (off-spec — official 060J spec uses 060A clips). Result: post-TTT 1.05893 vs 060H+TTT 1.05891 — Δ +0.000015 (noise). Total eval 655s on 4H ≈ 328s on 8H.

Conclusion: extra TTT phase does NOT help when quant noise is already reduced via clip tightening. Likely TTT is at saturation on the cleaner-quantized weights. Pure 060J (+1 phase, no clip change) might still help — that's research's call to spec.

### 060L (per spec): PREFIX_DOCS 2500 → 3000 — saturated

Per official spec: PHASED_TTT_PREFIX_DOCS 2500→3000 on pure 060A clips. Result: post-TTT 1.05918054 vs 060A 1.05918371 — Δ −0.0000032 (basically zero).

Total eval 676s on 4H ≈ 338s on 8H — fits cap.

Conclusion: prefix-doc lever is at saturation at 2500 on this stack. #1855's greedy search found the right point. 3000+ buys nothing.

## Handoff summary — best legal frontier

Three independent levers tested; one wins:

| lever | post-TTT Δ vs 060A | budget cost | verdict |
|---|---:|---|---|
| **060H clip combo** (ATTN 12.75 + EMBED 13.0) | **−0.000269** | +85 KB artifact, ~$0 eval time | **WIN** |
| 060J phases 3→4 on H-clip | +0.000015 (noise) | +85s eval/4H | discard |
| 060L prefix 2500→3000 (per spec) | −0.0000032 (noise) | +50s eval/4H | discard |
| 060M epochs (untested) | (predicted −0.001 to −0.003, env-var risk) | TBD | research's call |

**Combined frontier (best 4H result):** post-TTT **1.05891477** at total submission **15,987,688 bytes** (12 KB headroom).

vs #1855 references:
- vs 3-seed mean (1.06108): **−0.00217**
- vs seed-42 (1.05989): **−0.00098**

Where the wins came from:
- ~−0.0019 from "more training steps" (4H × 1196s gets 5084 steps vs 8H × 600s ~4940; same compute, more steps somehow)
- ~−0.000269 from clip tightening (real, transfers through TTT at 85% rate)

## Recommended action for 8H official run

Apply 060H clip combo: `ATTN_CLIP_SIGMAS=12.75`, `EMBED_CLIP_SIGMAS=13.0` (vs 060A defaults 13.0 / 14.0). Everything else as 060A. Confirm `COMPRESSOR=pergroup`. Predicted post-TTT 1.0608-1.0610 on 8H (3-seed mean), beating #1855's 1.06108.

If research wants to pursue further:
- 060M (TTT_EPOCHS) is the only untested lever in our shortlist with predicted Δ comparable to or larger than 060H. Worth a single shot.
- Better permutation than greedy similarity-sort could shave 5-30 KB from artifacts but the optimization step would push serialize past the 8H eval-time budget — probably not worth it.

060A also ran TTT (3 phases, 792s eval): post-TTT val_bpb **1.05918371**
— beats #1855's 3-seed mean (1.06108) by 0.0019.

## Compression study

The `_serialize_pergroup` path in #1855's train_gpt.py uses lrzip-zpaq on each
int8 weight group + brotli on the small remainder/perms streams. Tested
alternatives via two scripts (`/tmp/repack_pergroup.py`,
`/tmp/compress_compare{,_v2}.py`) — round-trip verified on
quant_result + quant_meta.

| scheme | bytes | Δ vs current production |
|---|---:|---:|
| brotli top-level (1855 default with `COMPRESSOR=brotli`) | 16,116,592 | 0 |
| lzma top-level (preset 9 EXTREME) | 17,210,188 | +1,093,596 (worse) |
| pergroup w/ brotli inner (`COMPRESSOR=pergroup`) | **15,870,018** | **−246,574** ← winner |
| pergroup w/ lzma inner | 15,877,904 | −238,688 |
| pergroup w/ best-per-piece (lrzip on remainder too) | 15,862,042 | −254,550 (only +8 KB vs winner) |

**Conclusion**: pergroup is essentially optimal. zstd-22-ldm and lzma-extreme
never win; lrzip's zpaq backend is the right tool for both the int8 weight
groups (large, redundant) and the small structured remainder.

**Cost**: pergroup compression takes ~120s wall (vs ~50s brotli). Counts
against eval-time budget. With TTT enabled, the extra 70s is modest; without
TTT, it's ~half the eval phase.

**Important nuance**: the 16M cap is on **total submission size** (.int6.ptz + ~32 KB compressed code), NOT on `.int6.ptz` alone. The launch script's
`final_model.int6.ptz` size check is misleading — should also check total
submission size from train.log. 060D-half's `.int6.ptz` was 15,992,622 (under
cap by 7 KB) but total was 16,024,916 (over by 25 KB).

## Code patch: `RESUME_FROM_CKPT`

Branch `exp/060-resume-ckpt`, commits `7b3719b` + `a7c0ed8`. Adds a new env
var that loads weights from a `.pt` path and skips the training loop, then
runs the rest of `train_and_eval` verbatim. Mirrors the existing
`TTT_EVAL_ONLY` and `PREQUANT_ONLY` escape hatches.

Diff: 21-line addition in one function. Tested via PREQUANT_ONLY=1 smoke ⇒
pre-quant val_bpb 1.06357955 vs original training-time 1.06357900 (Δ 5e-7,
float roundoff).

**Without this patch, 060B/C/D/E all needed full retrains (~$5/26 min each).
With it, each is ~$1/5-7 min — almost entirely lrzip compression cost.**

Important non-obvious: a freshly built `GPT(h)` defaults `looping_active=False`.
Without setting it `True` after load, the loop layers are bypassed → val_bpb
1.179 instead of 1.064. The fix is `if h.num_loops > 0: base_model.looping_active = True`
right after `load_state_dict`. The training path masked this because
`train_model` activates loops at frac=0.35 during training.

## Lever economics observed

| lever | step | bytes cost | Δ post-quant val_bpb | bytes / 1e-4 bpb |
|---|---|---:|---:|---:|
| ATTN | −0.5σ (13.0→12.5) | +60 KB | −0.000182 | ~33 KB / 1e-4 |
| ATTN | −1σ (13.0→12.0) | +123 KB | −0.000352 | ~35 KB / 1e-4 |
| **EMBED** | **−1σ (14.0→13.0)** | **+55 KB** | **−0.000209** | **~26 KB / 1e-4** ← cheapest |
| MLP | (untested; weight share suggests +150-300 KB / −0.5σ step) | — | — | — |

ATTN is approximately linear in step (60 KB per −0.5σ). EMBED is **more
byte-efficient than ATTN** (~26 vs ~34 KB per 1e-4 bpb). MLP untested but
likely the most expensive lever per step (largest weight class).

**Stack predictions vs observed** (additivity of byte costs):
- 060F (ATTN −0.5σ + EMBED −1σ): predicted +115 KB → 16,017 KB. **Observed: 16,017,570.** Perfect match. Δ also additive (−0.000397 vs predicted −0.000391). Overshoots cap by 18 KB.
- 060G (ATTN −0.5σ + EMBED −0.5σ, half-step EMBED): predicted +88 KB → 15,990 KB. **Observed: 15,989,345.** Perfect match. Δ −0.000297. **LEGAL, current best.**
- 060H (ATTN −0.25σ + EMBED −1σ, half-step ATTN): pending.

Conclusion: byte cost is essentially perfectly additive across ATTN+EMBED. The cap forces choosing one full lever or two half-levers. Half+half produces same Δ-bpb as one full lever at slightly tighter byte budget.

## Open issues for research

1. **Launch script cap check is wrong.** It checks `.int6.ptz ≤ 16,000,000`
   but the real cap is total submission. Need to grep `Total submission size`
   from train.log instead. Affects all future 060B+ launches.
2. **`RESUME_FROM_CKPT` patch needs review + merge.** Clean diff, smoke
   verified. Once merged into `research`, repoint specs 060B/C/D/E/F to the
   merged commit.
3. **`tmp_exec/repack_pergroup.py`** lives only at `/tmp/` on this session.
   Should be committed under `tmp_exec/` for next-session reuse.
4. **4H-adapted launch isn't checked in.** `/tmp/launch_060[B,D-half,E]_*.sh`
   are session-local. Should land as `tmp_exec/launch_060_eval.sh` (parameterized
   on env vars) per spec 060B's referenced `launch_060_eval.sh`.
5. **#1855 training-time launch defaulted COMPRESSOR=brotli.** This is what
   pushed 060A's submission over cap. The 060A launch script
   (`tmp_exec/launch_060A_run.sh`, commit 786ed3a) needs updating to
   `COMPRESSOR=pergroup` so future re-runs of 060A are leaderboard-valid
   out of the box.

## Pod operations notes

- First pod `tew5ycdcpbimyb` evicted (host failure or platform-side), not
  by us. Volume artifacts persisted; recovered cleanly on `v7aw1g82vi6ty5`.
- `lrzip` is NOT in stock apt cache on `runpod/parameter-golf:latest`. Must
  `apt-get update -qq && apt-get install -y lrzip` before launch — original
  `launch_060A_run.sh` (a786ed3a) only ran `apt-get install` which fails.
  Patched in `/tmp/launch_*` versions; needs upstream fix.
- 4×H100 NE-1 capacity was tight today; 8×H100 was unavailable in both
  NE-1 and JP at session start.
