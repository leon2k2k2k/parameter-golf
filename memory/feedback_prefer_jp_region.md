---
name: JP and NE-1 are both acceptable regions
description: User is happy with either JP-1 or NE-1; always try JP first, fall back to NE-1 silently on unavailability
type: feedback
---

Try AP-JP-1 first on every pod create. If JP is unavailable, fall back to US-NE-1 immediately — no need to ask the user, both regions are acceptable.

**Why:** Both regions have the same network volume layout (/workspace/parameter-golf/... paths identical). User confirmed "I am happy with both JP-1 and NE-1" — JP is preferred but NE-1 is an always-valid fallback, not a diagnostic vehicle.

**How to apply:** Pod create flow: try `--data-center-ids AP-JP-1`; on failure, retry with `--data-center-ids US-NE-1`. No other regions (IN/EU/CA/SEA) without explicit approval.
