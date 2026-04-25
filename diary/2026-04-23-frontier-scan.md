# 2026-04-23 — Frontier Scan (incremental)

Incremental scan from 2026-04-22T12:00:00Z.

## Main takeaway

The important new public signal is **PR #1779**: a **1.06421** tokenizer-disputed/likely-legal result on top of the #1736 CaseOps stack using a **frozen recurrent alpha/beta carry blend** plus the **#1767 LoRA warm-start-A** TTT recipe.

That matters because it is not a new direction for us. It is outside confirmation that the carry-blend family is real enough to beat our 1.06549 baseline publicly when composed with the strong CaseOps stack.

## What changed

- **#1779** becomes the best public tokenizer-disputed/likely-legal PR at **1.06421**, overtaking #1769.
- **#1780** packages a progressive recurrence schedule on an older SP8192 stack, but the idea is already in our research notes and the number is not competitive.
- **#1775** and **#1776** are clean but mostly compositional / reproduction-style submissions, useful as lineage evidence rather than as new levers.
- **#1774** introduces shared-specific attention to buy artifact budget for 12L + wider MLP, but the result is too weak for current priority.

## Research implication

The scan does **not** surface a new must-spec lever.

It does strengthen confidence in the **cross-layer carry / frozen alpha-beta** line we are already pursuing. The public frontier is now explicitly telling the same story our recent 031/032 internal work suggested:

1. learned recurrence carry is real
2. it stabilizes to structured nontrivial values
3. it can matter enough to move the full-stack record number

## Immediate priority

Keep treating carry-blend / alpha-beta recurrence as an active frontier thread rather than a speculative side branch. The new public result is a corroboration event, not a pivot.
