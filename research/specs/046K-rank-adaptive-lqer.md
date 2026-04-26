# Spec 046K — rank-adaptive LQER (per-tensor rank by residual norm)

**Slug:** `046K-rank-adaptive-lqer`
**Created:** 2026-04-27
**Status:** DRAFT — needs ~30-50 lines code
**Branch:** `exp/046-quant-repair`
**Commit:** TBD
**Parent:** `research/ideas/quant-repair.md` (Q-O), 046D results

## Hypothesis

Currently LQER applies RANK=4 uniformly to all top-K tensors. But residual
norms differ across tensors — `tok_emb` has a much larger residual than the
12th MLP fc tensor. **Allocating rank in proportion to residual contribution
should give better BPB-per-byte than uniform rank.**

046D showed RANK=6 uniformly was null and increasing TOP_K to 5/8 was null
because marginal tensors don't have enough residual structure to capture.
But that says nothing about whether redistributing the SAME total rank
budget across tensors would help.

Concretely: instead of 12 tensors × rank-4 = 48 total rank-units, try:
- tok_emb: rank-8 (it has the biggest residual)
- top 3 MLP fc: rank-6 each (next biggest)
- bottom 8 MLP fc: rank-3 each
- Total: 8 + 18 + 24 = 50 rank-units (similar bytes, better-allocated)

## Code change required

Modify LQER selection logic:
1. Compute residual norm `||W - W_q||_F` per tensor (already done for top-K)
2. Sort tensors by residual norm
3. Allocate rank proportionally — e.g., `rank_i = round(K * residual_norm_i / mean_residual_norm)` clamped to [1, 8]
4. Run LQER per-tensor with assigned rank
5. Pack with per-tensor rank metadata

~30-50 lines change.

## Arms

| Arm | Strategy | Notes |
|---|---|---|
| **046K-proportional** | rank ∝ residual norm, clamped [1, 8] | base implementation |
| **046K-emb-rank8** | tok_emb=8, all MLP fc=4 (boost emb only) | targeted bump |
| **046K-top3-rank6** | top-3 by residual = rank 6, rest = rank 3 | smart rebalance |
| **046K-budget-matched** | total rank budget = current 48 units, redistribute | strict same-bytes |

## Predicted outcome

- **Best case**: -0.001 to -0.002 BPB at same/lower bytes (rebalancing helps)
- **Realistic**: -0.0003 to -0.0008 BPB at same bytes
- **Pessimistic**: 0 (LQER is already at its Pareto)

## Acceptance

Reference = 046 verification quantized = 1.07467, size 15,953,718.

Per arm:
- **Win**: quantized < 1.0739 AND size ≤ 16,000,000
- **Same-Pareto**: quantized 1.0739–1.0760 AND size ≤ 16,000,000
- **Loss**: quantized > 1.0760 OR size > 16,000,000

## Risk

Low-medium. LQER is well-tested; just changing rank allocation. Worst case
is null (matches 046D findings).

## Why this might work

046D's null results tell us:
- Adding rank uniformly (RANK=6) doesn't help — diminishing returns on each tensor
- Adding tensors uniformly (TOP_K=5/8) doesn't help — marginal tensors have no juice

Both null because they spread budget UNIFORMLY. But the residual structure is
HIGHLY non-uniform (tok_emb residual >> 12th MLP fc residual). Concentrating
LQER budget where residual lives might extract more BPB per byte.

## Why this might NOT work

If LQER rank-4 already captures all the meaningfully-low-rank structure of
the highest-residual tensors, rank-6 or rank-8 just captures noise. We saw
RANK=6 uniform was null — bumping ONE tensor to rank-8 might also be null.

In that case, the budget freed by reducing low-residual tensors to rank-2/3
also doesn't recover anything (because their residuals were already small).
Net = null.

## Cost

~$4-5 (4 arms × $1 each + 30 min code).

## Decision tree

| Outcome | Next |
|---|---|
| Any arm wins ≥ -0.0005 | Adopt; consider stacking with 046I compression wins |
| All null | LQER allocation truly doesn't matter on this stack; close direction |
| Bytes increase without BPB win | Bug in implementation; fix |
