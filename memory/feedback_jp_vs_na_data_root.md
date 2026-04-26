# Feedback — JP and NE CaseOps path layout normalized

As of 2026-04-23, the JP volume was normalized to match the NE repo-local
CaseOps layout.

Canonical path root for the main repo on both NE and JP:
- `DATA_DIR=/workspace/parameter-golf/data`

Canonical CaseOps paths on both NE and JP:
- dataset root:
  `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`
- tokenizer:
  `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- val-bytes sidecar:
  `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin`

What changed:
- JP originally stored the full CaseOps tree under `/workspace/data/datasets/...`
- the tree was moved to `/workspace/parameter-golf/data/datasets/...` so the
  main repo path now matches NE

Rules:
- For the main `parameter-golf` repo, use
  `DATA_DIR=/workspace/parameter-golf/data` on both NE and JP.
- Do not switch to `DATA_DIR=/workspace` for CaseOps runs in this repo unless
  the volume is re-verified and documented otherwise.
- The `tput_test` checkout does not carry its own separate CaseOps tree; the
  main repo path is the relevant reference.
