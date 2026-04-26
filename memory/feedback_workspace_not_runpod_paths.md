---
name: Use /workspace paths, not /runpod
description: Default repo, data, and artifact paths should use /workspace; do not write specs with /runpod paths
type: feedback
---

Use `/workspace/...` paths in specs and execution commands by default. Do **not** write new specs with `/runpod/...` repo, data, or artifact paths unless a specific environment has been verified to require that exact mount.

For this project, the remembered default is:

- repo checkout under `/workspace/...`
- data under `/workspace/data` unless the execution environment says otherwise
- artifacts under `/workspace/runs/...`
- keep `TORCHINDUCTOR_CACHE_DIR` on `/tmp`, not on `/workspace` or `/runpod`

**Why:** Recent spec work accidentally used `/runpod/...` paths, but the current environment convention for this project should be `/workspace/...`. Path assumptions need to be pinned correctly in memory so future specs do not inherit the wrong mount layout.

**How to apply:** When drafting or freezing specs, replace `/runpod/...` command paths with `/workspace/...` and only deviate if the execution environment is explicitly re-verified first.
