# 2026-04-23 frontier scan 2

Incremental scan from `2026-04-23T03:30:00Z` to `2026-04-23T13:55:08Z`.

Main update: **#1787** is now the public likely-legal CaseOps leader at **1.06378**, ahead of **#1779 @ 1.06421** and below our **#1736 baseline @ 1.06549**. It is not clean because it inherits CaseOps, but it is a real frontier datapoint: 3 seeds, artifact/train/eval compliance, and a specific lever bundle (`Polar NS + MIN_LR floor + sparse attn gate + fused CE`) on top of the #1736 stack.

Other notable reads:

- **#1784** is clean and competent, but mostly composition: #1767 LoRA-TTT stack plus #1736 GatedAttn with support-path fixes. Watch-level, not a new frontier lever.
- **#1785** and **#1782** should not enter our clean frontier. Their claimed gain comes from an online byte-level PPM / n-gram-style eval memory over the validation stream. For our scan heuristics that lands in the banned eval-cache bucket even if the body argues "legal TTT".
- **#1788** has a non-record quant/training bundle (`QAT cooldown + INT4 MLP + NuMuon-lite`) but it is too far above frontier and too quant-heavy to prioritize now.
- **#1783** is still more of a wishlist bundle than a validated branch.

Actions taken:

- Wrote the detailed scan to `pr-analysis/frontier-scan/2026-04-23-2.md`.
- Updated `research/frontier-map.md` with **#1784** under **#1767** and **#1787** under **#1736**.
- Updated `research/frontier-state.json`.
- Added `research/ideas/polar-ns-sparse-gate-min-lr.md` as the one new idea note worth keeping from this batch.
