# 2026-04-23 — NE/JP CaseOps path report

Purpose: record the exact CaseOps asset locations used by execution so research
does not have to rediscover them.

## NE main repo layout

Canonical root:
- `/workspace/parameter-golf/data`

CaseOps dataset root:
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`

CaseOps tokenizer:
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`

CaseOps val-bytes sidecar:
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin`

Notes:
- The live NE pod confirmed this layout under the main repo checkout.
- The `tput_test` checkout does not have its own separate CaseOps tree.

## JP original layout

Before normalization, JP stored the same CaseOps tree at:
- `/workspace/data/datasets/fineweb10B_sp8192_caseops`

That made `034d` fail because the run resolved repo-local paths under:
- `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/...`

## JP normalized layout

On 2026-04-23, the full JP CaseOps tree was moved to match NE:
- from:
  `/workspace/data/datasets/fineweb10B_sp8192_caseops`
- to:
  `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops`

Verified after move:
- tokenizer exists at the NE-style path
- dataset root exists at the NE-style path
- `fineweb_val_bytes_000000.bin` exists at the NE-style path

## Operational guidance

For the main repo on both NE and JP, use:
- `DATA_DIR=/workspace/parameter-golf/data`

Execution should still verify these three paths before launch:
1. tokenizer model
2. CaseOps dataset directory
3. `fineweb_val_bytes_000000.bin`
