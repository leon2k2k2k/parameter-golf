# 2026-04-24 — Frontier Scan (incremental)

Incremental scan from the repo-local frontier state at `2026-04-24T02:13:46Z`.

## Main takeaway

This was mostly a bookkeeping scan, not a frontier-moving one.

The only new record-track PR worth carrying forward is **#1802**. It is clean and below merged SOTA at **1.07710**, but it does not threaten the live public frontier. Everything else in the delta is either non-record (`#1804`, `#1805`), draft/unverified (`#1803`), or still banned for planning purposes (`#1795`).

## What changed

- **#1802** enters the map as a clean Trunk A sibling off `#1493`. The lever is optimizer-side: Polar Express NS coefficients plus a warmdown floor for Muon on a global multi-phase TTT stack.
- **#1795** rebuilt after the reviewer pushback that closed `#1785`, but the core mechanism is unchanged: an online byte-level PPM mixture over already-seen validation bytes. That keeps it outside our frontier even though the submission is now cleaner and more explicit.
- **#1804** is actually useful despite being non-record. It packages a static byte-count audit for the `#1698` lineage and makes the byte-accounting dispute easier to discuss precisely.
- **#1805** is a real non-record research datapoint: compression-aware QAT on `#1493`. Interesting as quant-side training research, but not strong enough numerically and not on the right track for our current push.

## Research implication

Priority ordering does not change.

The public frontier is still:

1. clean line led by **#1756**
2. likely-legal tokenizer line led by **#1797**
3. byte-bug and pre-quant branches still disputed

`#1802` is a reminder that older legal-SP8192 stacks can still get under merged SOTA with optimizer polish, but it does not create pressure to redirect away from the current Trunk A frontier.
