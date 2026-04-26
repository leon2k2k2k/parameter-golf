---
name: Always send plots to Discord
description: Every plot generated must be sent to Discord immediately after creation, without being asked.
type: feedback
---

Always send every generated plot/image to Discord immediately after saving it — don't wait to be asked.

**Why:** User wants to see plots on Discord in real time; asking each time is friction.

**How to apply:** Any time a matplotlib (or other) plot is saved to disk, follow it immediately with a Discord send using the send-to-discord pattern. Include a brief caption describing what the plot shows.
