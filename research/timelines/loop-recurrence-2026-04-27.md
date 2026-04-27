# Layer-loop / depth-recurrence — public PR timeline (record track)

**Date:** 2026-04-27
**Scope:** record track (`track_10min_16mb/`) on `openai/parameter-golf`. Non-record PRs noted only when they carry the technique forward.
**Convention warning:** different PRs use different `NUM_LOOPS` semantics. In our current code (#1736 lineage), `NUM_LOOPS=N` means the loop band is traversed `N+1` total times (i.e. `NUM_LOOPS=2` = 3 passes through layers 3-5 → 17 virtual layers from 11 physical). Earlier PRs (#1334, #1394) call the same configuration "2-layer recurrence" or "2 passes" depending on whether they count the initial pass. I disambiguate inline as `(layers, total_passes)` whenever I quote a PR.

---

## 1. Origin — when did layer-looping first appear on the record track?

**PR #1204 — msisovic, 2026-04-01, MERGED.**
Title: "Record: ParallelResiduals + MiniDepthRecurrence — 1.1063 BPB."
This is the first record-track PR that actually shipped depth recurrence as part of a merged result. msisovic explicitly calls his variant **"Mini Depth Recurrence"** because earlier full-stack-recurrence attempts (PRs in the 100–600 range, all exploratory non-record) failed.

Mini-DR core findings (msisovic's own writeup):
- Repeating one layer helped, **two consecutive layers helped more**, three "was already losing to the step-time penalty" (on his stack).
- Sweet spot: **layers `4,5`** ("right around the U-Net hinge point").
- Repeating a single block "everywhere" was worse than reusing a small middle slice.
- **Delayed activation:** always-on recurrence reached ~1.1163; delayed (`RECUR_START_STEP=3000`) reached ~1.1153.
- **Untying the repeated MLPs** while keeping the rest of the recurrent block shared → another small win.

Origin config from the merged reproduction command:
```bash
RECUR_LAYERS=4,5 RECUR_START_STEP=3000 \
  REPEAT_UNTIE_MLP=full REPEAT_UNTIE_MLP_LAYERS=4,5 \
  PARALLEL_RESIDUAL=1 PARALLEL_START_LAYER=7
```

So the *first* loop config on the merged track was: **layers {4,5}, NL=1 extra pass (one repeat), step-based onset at 3000, untied MLPs in the loop band**.

Earlier non-record / contested submissions had recurrence too (PRs #79, #91, #103, #126, #268, #319, #341, #559…) but none of them landed as the SOTA.

---

## 2. Phase-by-phase config evolution on the record track

Each phase below names a representative PR and the lineage step it took. Bold numbers are claimed bpb.

### Phase A — origin (NL=1 extra, layers 4–5)
- **#1204 (msisovic, 2026-04-01) — 1.1063.** Loop {4,5}, NL=1 extra (2 passes total), step-based onset (`RECUR_START_STEP=3000`), untied MLPs in loop band, parallel residuals from L7. **First merged record with looping.**

### Phase B — productionized in clarkkev's stack (still {4,5}, NL=1 extra)
- **#1394 (clarkkev, 2026-04-05) — 1.08563 (5-seed).** Switches to "simpler implementation" of #1204. Body: "Loop layers 4-5 twice (while sharing params): the idea is from #1204, but this script uses a simpler implementation and loops twice rather than once." Same {4,5} band, NL=1 extra (2 passes), shared params. Adds SP8192 + GPTQ embeddings + SDClip. Becomes the trunk that everyone forks off.

### Phase C — extension to {3,4,5} (NL=1 extra)
- **#1331 (dexhunter, 2026-04-04) — 1.0900.** Extends band from {4,5} to {3,4,5}. NL=1 extra (2 passes total = "3-layer recurrence"). 14 virtual layers from 11 physical. Pairs with WD=0.095 + MLR=0.022.
  - Quote: "Repeating one layer helped, repeating two consecutive layers helped more, and repeating three was already losing to the step-time penalty" was msisovic's negative finding — #1331 reverses it by adding compensating WD/LR re-tuning.
- **#1285 (dexhunter, 2026-04-03) — 1.0912.** Earlier intermediate at {4,5} + NL=1 extra + WD synergy.

### Phase D — schedule and step-count tuning (X-Abhishek-X)
- **#1421 (X-Abhishek-X, 2026-04-06) — 1.0925.** Adds EMA tuning (0.9965). Same {3,4,5} band, NL=1 extra.
- **#1445 (X-Abhishek-X, 2026-04-07) — 1.0889.** First explicit move to **early-onset step 2000** (was 3000 in #1204, ~0.40+ in older runs). Combined with WARMDOWN_FRAC=0.72.
- **#1471 (X-Abhishek-X, 2026-04-08) — 1.0866.** Switches base tokenizer to SP8192. Same loop config: {3,4,5}, NL=1 extra, RECUR_START_STEP=2000.

### Phase E — extra pass (NL=2 extra, layers 4–5)
- **#1420 (abaybektursun, 2026-04-06) — 1.08309 (5-seed).** This is the **first NL=2 extra (3 passes total) on layers 4-5** on the record track. Config: encoder = `[0,1,2,3,4,5,4,5,4,5]`-style with layers 4-5 repeated 3 times total → 17 virtual from 11 physical.
  - Sweep result quoted: "Quadruple looping (19 virtual) was worse because the step count drops too far."
  - Frac sweep: `{0.30, 0.35, 0.40, 0.50}` on seed 1234. **0.35 won, 0.40 close.** "Below 0.35 the model doesn't get enough non-looped warmup and quality degrades." This is the *original* basis for `ENABLE_LOOPING_AT=0.35`.
- **#1450 (andrewbaggio1, 2026-04-07) — 1.08480 (5-seed).** Same NL=2 extra on layers 4-5, plus TMA megakernel for throughput.
- **#1437 (dexhunter, 2026-04-07) — 1.08091.** Stacks parallel residuals + 3-layer recurrence (now {3,4,5} again, 2 passes).

### Phase F — merged SOTA: {3,4,5}, NL=1 extra (2 passes), frac=0.35
- **#1493 (bigbag, 2026-04-09) — 1.0810 (3-seed). MERGED.**
  Config that became canonical: `loop_start=3 loop_end=5 num_loops=2 enable_looping_at=0.35` — but note bigbag's `num_loops=2` here means **3 passes total** (this is the convention our #1736 inherits). "3-layer depth recurrence (L3-5, activate at 0.35) — 17 virtual layers from 11 physical."
  This is where all our specs trace from.

### Phase G — schedule innovation: progressive activation
- **#1780 (wisebreadloaf, 2026-04-22) — 1.08061.** "Progressive recurrence." Replaces the single-step activation with a two-phase schedule:
  - Phase 1 at frac=0.35: **one extra recurrence pass.**
  - Phase 2 at frac=0.55: **full 3-layer recurrence.**
  Same loop band {3,4,5}, two activation events instead of one.

- **#1786 (sachinnchaudhary, 2026-04-23) — non-record ablation.** V1/V2/V3 ramps around frac=0.50:
  - V1: hard switch at 0.50.
  - V2: ramp 0.44 / 0.50 / 0.56.
  - V3: narrower ramp 0.485 / 0.500 / 0.515.
  - V3 narrowly beat V2 (1.09183 vs 1.09186 — basically tied). "Throughput tax from recurrence remains substantial once enabled." The ramps tested matter only for the same-frac transition, not earlier-frac.

### Phase H — extension to {3,4,5,6} (NL=1 extra)
- **#1678 (tashapais, 2026-04-16).** Title: "SP8192 + 4-Layer Depth Recurrence (loop_end=6)." Just bumps `LOOP_END` from 5 to 6 → loop band becomes {3,4,5,6}, NL=1 extra (2 passes), giving "19 virtual layers, 9 U-Net skips" (vs the 17/8 baseline). PR is open, **no 8×H100 number reported** as of Apr 26 — diff explicitly says "pending."

### Phase I — depth curriculum during training
- **#1756 (romeerp, 2026-04-20) — 1.06505 (3-seed).** Novel: **deterministic 1→3→4 recurrence-depth curriculum** after the loop activates at 35%. The loop band is still {3,4,5}, but the *number of passes through it* changes during training:
  - First third (post-activation): depth = 1 (no extra recurrence — base 11 layers only).
  - Second third: depth = 3 (3 total passes through the band).
  - Final third + eval + TTT: depth = 4 (4 total passes through the band).
  - Env vars: `TRAIN_LOOP_PHASE_DEPTHS=1,3,4`, `TRAIN_LOOP_PREWARM_DEPTHS=3,4`, `EVAL_LOOP_DEPTH=4`.
  Beats #1736 by 0.00044 bpb. Hypothesis: "teach the recurrent block to act like a scalable refinement operator, then evaluate at the deepest trained depth."
- **#1771 (bigbag, 2026-04-22) — 1.06513 (3-seed).** Composes the #1756 depth curriculum with #1736's CaseOps + GatedAttn stack.

### Phase J — pass-gating
- **#1697 (Buld1n, 2026-04-17) — 1.08120 (non-record).** **Pass-gated recurrence**: an extra recurrent attention gate (`recur_attn_delta`) so reused middle blocks are not exact repeats of the ungated path. Loop band still {3,4,5}, NL=1 extra. Onset sweep over `ENABLE_LOOPING_AT_STEP ∈ {1600, 2000, 2400, 2600, 2800, 3000}` — **2600 won** in this stack. (For our 4×H100 1200s budget that is ~0.50 frac, much later than our 0.35 default.)
- **#1766 (tashapais, 2026-04-22) — pending hardware.** Recur-Alpha (learned scalar carry init=0) on the #1736 stack — adapts what each pass of the loop sees rather than gating which passes run.

### Phase K — recurrent carry / blend (our submissions)
- **#1714 (Anakintano, 2026-04-18).** Origin of Recur-Alpha (learned per-block carry scalar, init=0). PR posted but no full-stack number — preserved as "future work."
- **#1779 (leon2k2k2k, 2026-04-22) — 1.06421.** **Frozen learned recurrent α/β carry**: Recur-Alpha trained to convergence then frozen as constants in the artifact. Loop config unchanged from #1736 (`{3,4,5}, NL=1 extra, frac=0.35`).
- **#1801 (leon2k2k2k, 2026-04-24) — 1.06287.** Updated frozen α/β + sparse attn-output gate. **Still loop {3,4,5}, NL=1 extra, frac=0.35.** Loop *config* unchanged — only what runs per pass changed.
- **#1797 (2026-04-24) — 1.06157.** SmearGate + LQER asym on #1787; same loop config.
- **#1787 (nprime06, 2026-04-23) — 1.06335.** Polar Express + sparse attn-gate; loop config unchanged.

### Phase L — non-record explorations (record-adjacent)
- **#1726 (krishs0404, 2026-04-18).** Explicit (LOOP_START, LOOP_END, ENABLE_LOOPING_AT) ablation, 1×H100 only — see §5 for table.
- **#1822 (Unwindology, 2026-04-25) — 1.17845 (non-record, far above baseline).** "BreadcrumbNode" — 9-layer U-Net with explicit "encoder–think–decoder" recurrence shape (3 enc, 3×3 think, 3 dec). Different architecture entirely; not a config of the looped layers but an entire arch.
- **#1828 (5en5e1, 2026-04-26) — 1.1169 (non-record).** "ETD Hybrid (3 enc + 3×3 think + 4 dec)" — same family as 1822, also off-trunk.
- **#1819 (DuoNeural, 2026-04-25) — pending.** "CTM-Golf: Triple Recurrence + Guided Hebbian Learning." On the #1736 stack, 3 looped layers — but pass count and exact band TBD; PR is prototype-only.
- **#1809 (PranavViswanath, 2026-04-24) — 1.0800 (3-seed).** SP8192 + Gram-NS + Polar Express + same {3,4,5} loop NL=1 extra. No new loop lever — just throughput recovery so more steps fit. Shows the loop config is locked across this trunk.

---

## 3. Score progression by loop configuration

| Date | PR# | Author | Loop band | NL convention | Total passes | Activation | bpb (mean) | seeds | merged? |
|------|-----|--------|-----------|---------------|--------------|------------|-----------|------:|---------|
| 2026-04-01 | 1204 | msisovic | {4,5} | NL=1 | 2 | step 3000 | **1.1063** | 3 | MERGED |
| 2026-04-03 | 1285 | dexhunter | {4,5} | — | 2 | step ~2000 | 1.0912 | 3 | merged-superseded |
| 2026-04-04 | 1331 | dexhunter | {3,4,5} | — | 2 | step ~2000 | 1.0900 | 3 | open |
| 2026-04-05 | 1394 | clarkkev | {4,5} | NL=1 | 2 | frac~0.35 | 1.08563 | 5 | MERGED |
| 2026-04-06 | 1421 | X-Abhishek | {3,4,5} | — | 2 | step 2000 | 1.0925 | 3 | open |
| 2026-04-06 | 1420 | abaybektursun | {4,5} | NL=2 | **3** | frac=0.35 | 1.08309 | 5 | open |
| 2026-04-07 | 1437 | dexhunter | {3,4,5} | — | 2 | frac~0.35 | 1.08091 | 5 | open |
| 2026-04-07 | 1445 | X-Abhishek | {3,4,5} | — | 2 | step 2000 | 1.0889 | 3 | open |
| 2026-04-07 | 1450 | andrewbaggio1 | {4,5} | NL=2 | 3 | frac=0.35 | 1.08480 | 5 | open |
| 2026-04-08 | 1471 | X-Abhishek | {3,4,5} | — | 2 | step 2000 | 1.0866 | 3 | open |
| 2026-04-09 | **1493** | bigbag | {3,4,5} | NL=2 (our conv) | **3** | frac=0.35 | **1.0810** | 3 | **MERGED SOTA** |
| 2026-04-10 | 1517 | RulinShao | {3,4,5} | — | 2 | step 2000 | 1.0632† | 3 | open |
| 2026-04-12 | 1572 | anthony-maio | {3,4,5} | NL=2 | 2 | frac=0.35 | 1.07974 | 3 | open |
| 2026-04-16 | 1678 | tashapais | {3,4,5,6} | NL=1 | 2 | frac=0.35 | pending | — | open |
| 2026-04-17 | 1697 | Buld1n | {3,4,5} | — | 2 | step **2600** | 1.08120 | 3 | open (non-rec) |
| 2026-04-18 | 1714 | Anakintano | {3,4,5} + α-carry | — | 2 | frac=0.35 | (concept) | — | open |
| 2026-04-20 | **1736** | dexhunter | {3,4,5} | NL=2 (=3 passes) | **3** | frac=0.35 | **1.06549** | 3 | open (our base) |
| 2026-04-20 | 1756 | romeerp | {3,4,5} | curriculum 1→3→4 | 1→3→**4** | frac~0.35 | **1.06505** | 3 | open |
| 2026-04-22 | 1766 | tashapais | {3,4,5} + α-carry | NL=2 | 3 | frac=0.35 | pending | — | open |
| 2026-04-22 | 1771 | bigbag | {3,4,5} | curriculum 1→3→4 | 4 | frac~0.35 | 1.06513 | 3 | open |
| 2026-04-22 | 1779 | leon2k2k2k | {3,4,5} + frozen α/β | NL=2 | 3 | frac=0.35 | 1.06421 | 3 | open |
| 2026-04-22 | 1780 | wisebreadloaf | {3,4,5} | progressive | 2→3 | 0.35→0.55 | 1.08061 | 3 | open |
| 2026-04-23 | 1786 | sachinnchaudhary | {3,4,5} | — | 2 | ramp ~0.50 | 1.09183 | 1 | open (non-rec) |
| 2026-04-24 | 1801 | leon2k2k2k | {3,4,5} + frozen α/β + sparse-gate | NL=2 | 3 | frac=0.35 | 1.06287 | 3 | open |
| 2026-04-25 | 1809 | PranavViswanath | {3,4,5} | NL=2 | 3 | frac=0.35 | 1.0800 | 3 | open |
| 2026-04-25 | 1812 | EthanNing | {3,4,5} | NL=2 | 3 | frac=0.35 | 1.0729 | 3 | open |

† PR #1517's 1.0632 uses 18-epoch *pre-quant* TTT — disputed, listed for completeness only.

---

## 4. Interaction notes

**Loop × throughput tax.** The dominant single empirical fact across PRs: every additional loop pass *and* every wider loop band *and* earlier activation costs steps. Every author who tried to push these levers reports the same trade. #1420 quote on NL=4: "quadruple looping (19 virtual) was worse because the step count drops too far." #1726 quote on heavy reuse {2..7}: "+0.163 BPB due to throughput loss: only 451 steps vs 568." Our local memory confirms (~+56% per-step tax post-activation on 4×H100).

**Loop × TTT.** Two TTT regimes both compose with the canonical Loop345/NL=2:
- Score-first sliding TTT (#1493 uses SGD lr=0.005, 3 epochs, cosine).
- Phased multi-stage TTT (#1626 → #1736 → #1779). The eval still runs the loop at full depth.
- #1756 explicitly evaluates *and TTTs* at depth 4 (the deepest trained depth). This is the closest the public record has come to "loop depth at eval ≠ loop depth at train."

**Loop × quantization.** No interaction reported. #1394 quote: "All 66 layers at int6 simultaneously" — the looped block is quantized once, used 3× in forward. GPTQ is per-physical-layer.

**Loop × parallel residuals.** Always orthogonal. From #1493 onward, every clean loop submission combines `LOOP_START=3 LOOP_END=5 NUM_LOOPS=2` with `PARALLEL_START_LAYER ∈ {7, 8}`. No PR has touched the boundary between the two.

**Loop × per-pass gating.** #1697 (pass-gated, recur_attn_delta) and #1714/#1766 (Recur-Alpha) and our #1779/#1801 (frozen learned α/β) all add a *single dedicated parameter (or 3, or 12) per loop block* to differentiate passes. None has so far reshaped the loop band. Loop band + NL is a separate axis.

**Loop on/off ablations.** Cleanest public on/off:
- #1764 (gmn0105, 2026-04-21) — "non-record no-looping SOTA-stack submission scaffold." Same #1493/#1736 stack with `NUM_LOOPS=0`. No bpb claim in our state file (not run as a comparison submission).
- #1726 (krishs0404) — five-variant sweep at 1×H100, 10-min budget. Baseline 1.4689 vs minimal-reuse {5,6} 1.4750. The on/off magnitude is enormous on the 1×H100 budget, but the relative ordering is the public signal.

---

## 5. Ablations: tested vs assumed in the public record

### Tested explicitly (with a number, public)

| Lever | Variants tested | Best | Source |
|-------|-----------------|------|--------|
| LOOP_START | 1, 2, 3, 5, 7 | 3 | #1726 (1×H100) |
| LOOP_END | 4, 5, 6, 7, 10 | 5 (with 6 pending in #1678) | #1726, #1678 |
| Loop band size | 1 layer, 2 layers, 3 layers, 4 layers, 6 layers | 3 layers ({3,4,5}) | #1204 (1-2-3 sweep), #1726 (heavy band killed by throughput) |
| NUM_LOOPS (extra passes) | 1, 2, 3 (for layers 4-5) | NL=2 (3 passes) | #1420 sweep |
| Activation timing (frac) | 0.15, 0.30, 0.35, 0.40, 0.50, 0.55 | 0.35 hard switch | #1420, #1726, #1786 |
| Activation timing (step-based) | 1600, 2000, 2400, 2600, 2800, 3000 | 2600 (in pass-gated stack) / 2000 (in clean stack) | #1697, #1445 |
| Schedule shape | hard switch, 2-phase progressive, V2 ramp, V3 ramp, depth curriculum | depth-curriculum (#1756) > 2-phase (#1780) > V3 ramp (#1786) > hard switch | #1756, #1780, #1786 |
| Untying within loop band | full untie, MLP-only untie, full share | full share (after #1394 simplification) | #1204 → #1394 |
| Per-pass differentiation | recur_attn_delta gate (#1697), Recur-Alpha scalar (#1714), frozen α/β (#1779/#1801), iter embeds (our spec 045) | frozen α/β at our scale; pass-gating untested at our stack | #1697, #1714, #1779, #1801 |
| Heavy reuse {2,..,7} | tested, killed by throughput | — | #1726 |
| Loop band shape | contiguous only | — | (no PR has tried non-contiguous) |
| Loop {5,6} only (band shifted late) | +0.006 over baseline | — | #1726 |

### Assumed but never publicly ablated on the modern stack

- **NL=4 / 5+** on the {3,4,5} band on the #1736 trunk. #1420's NL=4 negative result was on the {4,5} band, with 2026-04-06 throughput (no Polar Express, no Gram-NS, no fused softcap-CE). Our local 041N tested NL=4 on the *shrunk* {4,5} band and lost in noise; the {3,4,5} NL=3 / NL=4 is **untested in the public record except via the #1756 curriculum (which reaches depth 4 only at the end)**.
- **Non-contiguous loop bands** (e.g. {3,5}, {2,4,5}, {3,5,7}). Zero public submissions.
- **Asymmetric encoder/decoder shape.** Every PR that publishes its index list uses a symmetric encoder + decoder (`enc=[0,..,5,3,4,5,3,4]`, `dec=[5,3,4,5,..,10]`). Nobody has published, e.g., `enc=[0..5,3,4,5]`, `dec=[5,6..10]` (encoder-only recurrence) or vice versa.
- **Per-iteration MLP variation.** msisovic's 1204 untied MLPs in the loop band but kept attention shared; #1394 collapsed back to fully shared and nobody re-explored the MLP-untie path on top of {3,4,5}/SP8192.
- **Different parameter sets per pass.** No record-track PR shipped per-pass parameters (small LoRA-per-pass etc.) — closest is #1697's single per-band gate.
- **Loop band that *changes* during training** (e.g. start as {3,4,5}, expand to {2,3,4,5,6} late). Zero public submissions.
- **Step-function loop deactivation** ("turn loop off in the last 5% to consolidate"). Zero public submissions.
- **Loop *width* (band) curriculum** analogous to #1756's depth curriculum. #1756 keeps the band fixed and varies depth; the dual ("keep depth fixed and grow the band") has not been tried.

---

## 6. What our local 04* runs already covered

So I don't repeat anything we've already screened. Cross-referenced against `research/specs/04*` and `runs/04*`:

| Lever | Public state | Our screen |
|-------|--------------|------------|
| Loop band {4,5} (shrunk) at NL=2/3 | 041G/041H/041I/041K/041N | #1420 found NL=4 on {4,5} bad; we extended to NL=4 on shrunk band — also bad. |
| frac=0.20 (early activation) | #1726 says 0.15 hurts; #1420 says <0.35 hurts on its stack | 043A on canonical {3,4,5}: +0.00096 vs baseline (small loss). |
| frac=0.25 with NL=3 on shrunk band | not in public | 041L: 1.06615 (worse than 041K) |
| LR slope shock at frac=0.20 | not in public | 042A/043B: net negative; killed |
| Activation in MLP middle layers (penalized_tanh, tanh, leaky 0.3) on the loop band | not in public | 039bA/B/C |
| Iter embeddings | not in public | spec 045 arm A2 (after optimizer-bug fix) |
| MLP-only loop (skip attn after pass 0) | not in public | spec 045 arm B |
| 1/L scale init in loop band | not in public | spec 045 arm C — best of the three so far (1.06479) |
| Per-pass resid_mix (`[num_passes, num_looped, 2, dim]`) | not in public | spec 045 arm H |
| Palindrome (ABBA-style) loop traversal | not in public | spec 045 arm I (running) |
| LR scaling 1/L on grad of looped params | not in public | spec 045 arm G (inert) |
| Depth curriculum at our stack | #1756 / #1771 confirmed gain | not yet specced — clear gap |
| Progressive 2-phase activation (#1780) | confirmed | not yet specced — clear gap |
| Pass-gated recurrence (#1697 recur_attn_delta) | confirmed (non-record) | not yet specced — clear gap |
| Loop band {3,4,5} with NL=3 (4 passes) | not in public | 045 Arm F is the closest; ran with optimizer bug, results unreliable |
| Asymmetric encoder/decoder loop count | not in public | not specced — clear gap |
| Loop band shrinkage *late* in training (depth curriculum 4→3→1) | not in public | not specced |

---

## 7. Configurations not yet tried — opportunities

Six concrete, prioritized opportunities. Each has (a) exact config, (b) plausible delta with reasoning, (c) one-liner vs code change, (d) cost/risk.

### Opp 1 — Adopt the #1756 depth curriculum, isolated on our #1736 base
- (a) `TRAIN_LOOP_PHASE_DEPTHS=1,3,4 TRAIN_LOOP_PREWARM_DEPTHS=3,4 EVAL_LOOP_DEPTH=4`. Loop band unchanged ({3,4,5}). No code change to forward; **light code change** to scheduler reading those env vars (or port from #1756).
- (b) **Plausible Δ ≈ −0.0004 to −0.0010.** #1756 alone got 1.06505 vs 1.06549 baseline (−0.00044). #1771 stacked it onto CaseOps + GatedAttn and landed 1.06513 (−0.00036 vs 1.06549). This lever is the only one in the public record that has demonstrably moved a competitive 3-seed mean below #1736 without disputed mechanism.
- (c) Code change (forward pass needs depth-aware index list at runtime).
- (d) ~$18 (8×H100, 3 seeds, full 600s). Risk: low. Already publicly verified twice (#1756, #1771).

### Opp 2 — Progressive activation (Phase G) on {3,4,5} with NL=3 endpoint
- (a) Two activation events: `ENABLE_LOOPING_AT=0.35` activates depth=2 (NL=1 extra), then a second event at 0.55 activates depth=3 (NL=2 extra). On the canonical band, not the shrunk one.
- (b) **Plausible Δ ≈ −0.0005 to −0.0015.** Logic: #1780's progressive saved ~−0.005 over the matched non-progressive baseline at the SP8192-only stack. Adding the third pass (NL=3) at the second event combines #1780's idea with #1420's NL=2-extra result, both already verified separately. The throughput tax of the third pass only kicks in after frac=0.55 (not 0.35), buying a much higher step count than a single-event NL=2-extra activation.
- (c) Same code as Opp 1's depth scheduler — natural overlap.
- (d) ~$18. Risk: medium. Has not been publicly tested at NL=3 endpoint.

### Opp 3 — Pass-gated recurrence (recur_attn_delta) on top of our current frozen α/β stack
- (a) On our #1801 base, add `RECUR_ATTN_GATE=1 RECUR_ATTN_GATE_SCALE=0.5` (per #1697's API). Loop {3,4,5}, NL=2 (3 passes), frac=0.35 unchanged.
- (b) **Plausible Δ ≈ −0.0003 to −0.0010.** #1697 on its own (non-record) reached 1.08120; the *delta* it claims over the matched non-gated SP8192 baseline is in the public README. The mechanism is orthogonal to frozen α/β: α/β controls the residual blend, recur_attn_delta controls per-pass attention parameters.
- (c) Code change: small (add gate parameter inside attn block, only active when `looping_active and pass_idx > 0`).
- (d) ~$25 (4×H100 mini + 8×H100 official). Risk: medium-high — interacts with our frozen α/β; need a clean ablation arm.

### Opp 4 — NL=3 on the canonical {3,4,5} band, baseline schedule
- (a) `LOOP_START=3 LOOP_END=5 NUM_LOOPS=3 ENABLE_LOOPING_AT=0.35` — i.e. 4 passes through layers 3,4,5 = 23 virtual layers. **No public PR has reported a 3-seed mean for this exact configuration.** #1420 dismissed NL=4 on {4,5} based on throughput cost; #1756 reaches depth 4 only via curriculum. Plain NL=3 on {3,4,5} as a static config is an empirical hole.
- (b) **Plausible Δ: ambiguous. Could be −0.0005 or +0.0010.** The math: 5–10% step-count loss from added pass × ~+5 layer-equivalents extra compute per token. On our local data the same lever on the shrunk {4,5} band gave 041K = 1.06563 (small win over 041H). Whether the canonical band absorbs the same lever cleanly is unknown.
- (c) **One-liner** (`NUM_LOOPS=3`).
- (d) ~$3.50 4×H100 mini, ~$18 official if mini wins. Risk: low. This is the single lowest-cost open ablation.

### Opp 5 — Asymmetric encoder/decoder loop depth
- (a) `ENC_NUM_LOOPS=2 DEC_NUM_LOOPS=3` (or vice versa) on the {3,4,5} band — encoder pass count differs from decoder pass count. Nothing in the public record explores this.
- (b) **Plausible Δ ≈ −0.0003 to +0.0005, very uncertain.** Theoretical motivation: the encoder builds representations, the decoder applies them via U-Net skip-connections — these jobs aren't symmetric and don't necessarily benefit equally from extra passes. Untested anywhere on the record track.
- (c) Code change: medium. Need to split `num_loops` into enc/dec versions in the index list builder + recur_alpha/iter_embeds bookkeeping. ~30 lines.
- (d) ~$30 across 4 mini arms (the 2×2 grid of {2,3} × {2,3}). Risk: medium — exploratory, no prior result to anchor on.

### Opp 6 — Loop band {3,4,5} with NL=2 *and* a late-phase band shrink to {4,5}
- (a) Train at full {3,4,5} band for 0.35 ≤ frac ≤ 0.85, then drop layer 3 from the loop band for the final 15% — i.e. simulate "decoder-side specialization" without changing the artifact. Inverse of Opp 5.
- (b) **Plausible Δ ≈ −0.0003 to +0.0005.** Logic: the {4,5} band has known per-loop-step quality cost vs {3,4,5} (our memory + #1726), but it's also cheaper. Reclaiming throughput in the *warmdown* phase, where LR is low and gradient info is low-signal, *might* let more EMA-stabilized steps fit. Pure speculation.
- (c) Code change: small, equivalent to a runtime band selector reading `LOOP_BAND_SHRINK_AT=0.85`.
- (d) ~$18 official + $3.50 mini. Risk: medium.

---

## Summary table

(Same as §3 — see above.)

## Key findings

1. **The {3,4,5} band has been dominant since 2026-04-04.** Every record-track submission since #1331 uses {3,4,5} (or its NL=2-extra variant {4,5}-with-3-passes from #1420). No PR has successfully moved the band on the modern SP8192 trunk.
2. **NL=2 (i.e. 3 passes) is the public optimum.** #1420 sweep killed NL=4. #1726 sweep reaffirmed that wider bands lose to throughput. NL=3 on {3,4,5} as a *static* config is an empirical hole — only #1756 reaches depth 4 and only via curriculum.
3. **frac=0.35 has been baseline since #1420.** Three independent sweeps (#1420, #1726, #1697 step-based, #1786 ramp) all converge near 0.35–0.50. Nobody has cleanly beaten 0.35 on the modern record stack.
4. **The depth curriculum (#1756) is the only schedule innovation that beat #1736 on a 3-seed mean.** Two PRs replicated it (#1771 = 1.06513). Yet it's not in our research stack — that's a real gap.
5. **Per-pass parameter additions are the recurring composer's lever.** Recur-Alpha (#1714, #1766), frozen α/β (#1779), pass-gating (#1697), iter embeds (our 045A2), per-pass resid_mix (045H), MLP-only-from-pass (045B). The public record has not yet tried *combining* the depth curriculum with any of these.

---

## Configs that have been tried (exhaustive, public record)

(Format: `(layer-set, total_passes, schedule)`)

- `({4,5}, 2, step 3000)` — #1204
- `({4,5}, 2, frac~0.35)` — #1394
- `({4,5}, 3, frac=0.35)` — #1420, #1450
- `({3,4,5}, 2, step 2000)` — #1331, #1421, #1445, #1471, #1517, #1437, #1572
- `({3,4,5}, 2, frac=0.35 hard)` — #1493 (MERGED), #1736, and ~all subsequent record submissions
- `({3,4,5}, 2, frac=0.35 + freeze last)` — #1626 family TTT
- `({3,4,5}, 2, step 2600)` — #1697 (pass-gated stack)
- `({3,4,5}, 2, ramp 0.50±0.015)` — #1786 V3
- `({3,4,5}, 2, progressive 0.35→0.55)` — #1780 (NL=1 extra → NL=2 extra)
- `({3,4,5}, curriculum 1→3→4, frac=0.35)` — #1756, #1771
- `({3,4,5,6}, 2, frac=0.35)` — #1678 pending
- `({2..7}, 2, frac=0.35)` — #1726 (killed by throughput)
- `({5,6}, 2, frac=0.35)` — #1726 (small +0.006 cost, surprisingly competitive)
- `({1..4}, 2, frac=0.35)` — #1726 (+0.038)
- `({7..10}, 2, frac=0.35)` — #1726 (+0.049)
- `({3,4,5}, 2, frac=0.15)` — #1726 (+0.050)

## Configs explicitly NOT yet tried on the current (#1736) stack

Flagged: **(public gap)** = nobody has tried it; **(local gap)** = our 04* runs haven't covered it either; **(local-covered)** = our runs have it.

- **NL=3 on canonical {3,4,5}, static (no curriculum), frac=0.35.** (public gap, local gap)
- **NL=2 on {3,4,5} with depth-curriculum 1→3→4.** (public: #1756/#1771 yes, local gap — never specced on our exp branch)
- **NL=2 with progressive activation 0.35→0.55.** (public: #1780, local gap)
- **NL=1+2-extra hybrid (encoder NL=1, decoder NL=2).** (public gap, local gap)
- **NL=1 on {3,4,5} (i.e. just one extra pass = 2 total passes through band) — actual ablation on our stack.** Was the canonical config in #1493 originally; we never re-tested it locally on #1736 stack as a "loop turned down" baseline. (local gap)
- **{2,3,4,5}, NL=2, frac=0.35.** Wider band, same depth. (public gap, local gap)
- **{3,4,5,6}, NL=2, frac=0.35.** #1678 pending — not run on our stack. (local gap)
- **Non-contiguous {3,5} or {2,5}, NL=2.** (public gap, local gap)
- **Pass-gated recurrence (#1697 recur_attn_delta) on our #1736 stack.** (public-confirmed non-record, local gap)
- **Loop band shrinkage in the last 10–15% of training.** (public gap, local gap)
- **Loop band growth (start small, grow late).** Dual of the depth curriculum. (public gap, local gap)
- **Per-pass MLP untying** (msisovic's #1204 lever, dropped in #1394; never restored). (public gap on modern stack, local gap)
- **Iter embeds + 1/L init + depth curriculum stacked.** (local arm 045D / E / F partially explored but with optimizer bug; clean rerun pending and never specced as a 3-arm ablation)
- **Loop only at eval** (train without loop, only enable for eval/TTT). (public gap, local gap)

---

## File pointers

- Local baseline code (Loop345/NL=2/frac=0.35): `/home/claude-user/ai-workspace/projects/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py` (lines 281–284, 1407–1430).
- Loop-band activation screen (our 039b series): `research/specs/039b-loop-band-activation-screen.md`.
- Shrunk-band 041 sweep: `research/specs/041L-…`, `041M-…`, `041N-…`.
- Our latest spec (loop layer improvements: A=iter embeds, B=MLP-only, C=1/L init, D=combined, etc.): `research/specs/045-loop-layer-improvements.md`.
- Frontier state machine-readable: `research/frontier-state.json`.
- Frontier dependency map (human-readable): `research/frontier-map.md`.
