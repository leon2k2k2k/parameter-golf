# 060A: PR #1855 port — research baseline (1-seed validation)

**Spec:** [`research/specs/060A-1855-port-baseline.md`](../../../research/specs/060A-1855-port-baseline.md)
**Source:** [PR #1855](https://github.com/openai/parameter-golf/pull/1855) (codemath3000) @ commit `1e43966`, branch `codemath3000:submission/sp8192-lqer-bos-smear-fix-9hp-stack`.
**Purpose:** Pure port of #1855 as the new research baseline for the spec 060 family. Files in this directory are **verbatim** from #1855 (no modifications); spec 060B+ will fork from this baseline to add quant-repair / deploy-time levers.

This is **not a submission** — it is a research baseline. The actual leaderboard credit belongs to PR #1855 (codemath3000). See `README.md.original` for the original PR text.

## What is here

- `train_gpt.py` — verbatim from PR #1855
- `lossless_caps.py`, `prepare_caseops_data.py` — CaseOps tokenizer pipeline (verbatim, originally from PR #1729)
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — verbatim
- `requirements.txt` — verbatim (note: requires `lrzip` system package)
- `README.md.original`, `submission.json.original` — codemath3000's original docs preserved

## Why we ported it

PR #1902 (cocohearts leaderboard) accepted #1851 (BOS-fix on #1797) + #1868 (3-seed reproduction) at val_bpb 1.06128 / 1.06145, and excluded #1855 only on significance grounds (p=0.325). Our prior research line built on #1797 directly and was placed under "validity/provenance" cloud per cocohearts. Spec 060A re-anchors our research baseline on the cocohearts-accepted #1855 chain to give us a clean foundation for the 060B+ quant-repair stacking.

## Stack additions (post-060A, in spec 060B–E)

- **060B**: 046B-tight SDClip (MLP=11.5/ATTN=12.5/EMBED=14.5)
- **060C**: 046L deploy-time quant repair (eval-side, free)
- **060D**: 046G-tighter SDClip (more aggressive; needs additional compression headroom)
- **060E**: full stack (060B + 060C + best of 060D)

## Credit

@codemath3000 — full stack (PR #1855)
@aquariouseworkman — BOS-fix (PR #1851)
@Christopher-Lee-McClendon — 3-seed reproduction (PR #1868)
@dexhunter — base (PR #1797)
@nprime06 — base-base (PR #1787)
@romeerp — CaseOps (PR #1729)
