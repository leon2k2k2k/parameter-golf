# Feedback: 8H full runs are 600s

Rule:
- In this repo's current execution conventions, a full `8×H` run is `600s`, not `1200s`.
- Do not carry over the `20min` assumption from `4×H` lines onto `8×H` specs.
- For `8×H` promotion specs, use `MAX_WALLCLOCK_SECONDS=600` unless the user explicitly asks otherwise.
