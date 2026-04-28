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
| 060E | EMBED_CLIP_SIGMAS 14.0→13.0 | (running) | — | — | — |

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

| lever | one −0.5σ step costs (bytes) | gives (Δ post-quant val_bpb) | bytes / 1e-4 bpb |
|---|---:|---:|---:|
| ATTN | +60 KB / step | −0.00018 / step | ~333 KB |
| EMBED | (060E running) | (060E running) | — |
| MLP | (untested; likely +150-300 KB / step based on weight share) | — | — |

Linear in ATTN so far. Need EMBED measurement (060E) to rank levers.

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
