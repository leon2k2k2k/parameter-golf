# Feedback — Pod specs must live on a remote execution branch

If a spec is meant to be run on a pod, the needed spec/code must be on the exact remote branch the executioner will fetch.

Rules:
- Do not treat a repo-root local spec edit as pod-ready.
- Do not rely on the current research branch for pod execution unless that exact branch is the intended execution branch.
- For pod-run specs, create or update the exact `exp/...` branch, commit there, and push there.
- When reporting readiness, distinguish clearly between:
  - local/root-tree research updates
  - real remote execution branches
