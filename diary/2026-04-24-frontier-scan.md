# 2026-04-24 — Frontier Scan (incremental)

Incremental scan from the last repo-local frontier report.

## Main takeaway

The important miss from the earlier same-day scan was **DexHunter's new PR #1797**. It is now the best public **tokenizer-disputed / likely-legal** result at **1.06157**, ahead of both **#1787 @ 1.06335** and **your #1801 @ 1.06287**.

That does not change the clean frontier. It does change the public CaseOps frontier: the likely-legal bucket moved again, and it moved by almost four millibpb relative to `#1736`.

## What changed

- **#1797** is the new public likely-legal CaseOps leader. Read it as a composition result on top of `#1787`, not as a single isolated lever.
- **#1801** still matters. It is another below-baseline frozen-carry confirmation on your line, just not the current bucket leader after `#1797`.
- **#1787** itself improved relative to the earlier report and should now be treated as **1.06335**, not **1.06378**.
- **#1796** is the only notable non-CaseOps novelty in the batch: a TokenMonster-derived tokenizer with explicit metadata-driven byte accounting. Interesting, but not strong enough numerically to redirect effort.
- **#1795** remains out-of-bounds for frontier planning. The mechanism is still an online byte-level PPM memory over validation bytes, so for us it stays in the banned eval-cache family.

## Research implication

This scan is mostly a **frontier refresh**, not a new-idea event.

The live story is:

1. the public likely-legal CaseOps line still has room below `#1736`
2. both your frozen-carry line and DexHunter's smear/LQER composition can cash that room in
3. none of the new clean PRs changed our priority ordering

## Unclassified lineage

- **#1796** (`Scylla`) stays outside the main dependency tree for now. It is record-track and likely legal, but its lineage is tokenizer-first rather than clearly attached to Trunk A/C/D.
