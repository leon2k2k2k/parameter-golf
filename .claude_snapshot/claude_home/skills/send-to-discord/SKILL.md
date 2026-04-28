---
name: send-to-discord
description: Send a message or file TO THE USER on Discord. This posts to the Discord channel — the user sees it. Do NOT use this to talk to Nova. To talk to Nova internally, use work-with-nova instead.
tools: Bash
---

# Send to Discord (Message the User)

Send messages and files **to the user's Discord channel**. The user will see these messages in Discord.

**This is NOT for talking to Nova.** If you want to ask Nova a math question and get a response back, use the `work-with-nova` skill (`openclaw agent --agent nova --message "..."`).

**No OpenClaw needed.** This calls the Discord API directly via Python `requests`.

## Quick Start

### Send a message
```python
import requests, os
os.environ['NO_PROXY'] = 'discord.com'
os.environ['HTTPS_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''

TOKEN = os.environ["DISCORD_BOT_TOKEN"]
CHANNEL = "1474618189806833745"
URL = f"https://discord.com/api/v10/channels/{CHANNEL}/messages"

r = requests.post(URL,
    headers={"Authorization": f"Bot {TOKEN}", "Content-Type": "application/json"},
    json={"content": "From Claude Code: your message here"},
    timeout=15)
print(r.status_code)
```

### Send a file (PDF, image, etc.)
```python
import requests, os
os.environ['NO_PROXY'] = 'discord.com'
os.environ['HTTPS_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''

TOKEN = os.environ["DISCORD_BOT_TOKEN"]
CHANNEL = "1474618189806833745"
URL = f"https://discord.com/api/v10/channels/{CHANNEL}/messages"

pdf_path = "/home/claude-user/CFTs-and-RH/run-output/5A/5A-Formulas.pdf"
with open(pdf_path, 'rb') as f:
    r = requests.post(URL,
        headers={"Authorization": f"Bot {TOKEN}"},
        data={"content": "From Claude Code: 5A analysis PDF"},
        files={"files[0]": (os.path.basename(pdf_path), f, "application/pdf")},
        timeout=30)
print(r.status_code)
```

## Important Rules

1. **Always prefix messages with "From Claude Code:"** so the user knows it's from Claude Code, not Nova
2. **Bypass the proxy** — must set `NO_PROXY=discord.com` and clear `HTTPS_PROXY`/`HTTP_PROXY`, otherwise SSL fails
3. **No staging needed** — unlike the old openclaw method, you can send files directly from any path

## How It Works

Uses Nova's Discord bot token to POST directly to the Discord REST API:
- **Endpoint:** `https://discord.com/api/v10/channels/{channel_id}/messages`
- **Auth:** `Authorization: Bot {token}`
- **Text:** JSON body `{"content": "message"}`
- **Files:** Multipart form with `files[0]` field

That's it. No openclaw gateway, no LLM relay, no media staging.

## Config

- **Bot token:** Nova's Discord bot token (in the code above)
- **Channel:** `1474618189806833745`
- **Bot username:** `nova` (messages appear as coming from Nova)

## Usage Patterns

### After pipeline completion
```bash
send-to-discord 5A
# Sends run-output/5A/5A-Formulas.pdf with default message
```

### Custom message only
```bash
send-to-discord -m "Analysis complete, check the results"
```

### Custom file
```bash
send-to-discord /path/to/any/file.pdf
```

### In Python scripts
```python
NO_PROXY="discord.com" HTTPS_PROXY="" HTTP_PROXY="" python3 << 'PYEOF'
import requests, os
# ... (see Quick Start above)
PYEOF
```

## Inline Bash One-Liner (message only)

```bash
NO_PROXY="discord.com" HTTPS_PROXY="" HTTP_PROXY="" python3 -c "
import requests
r = requests.post('https://discord.com/api/v10/channels/1474618189806833745/messages',
    headers={'Authorization': f'Bot {os.environ["DISCORD_BOT_TOKEN"]}',
             'Content-Type': 'application/json'},
    json={'content': 'From Claude Code: your message'}, timeout=15)
print(r.status_code)
"
```

## Integration

This skill replaces `send-doc-via-nova`. It is called at the end of:
- `mtc-analysis-pipeline` (Step 8: deliver PDF)
- `batch-analyze` (Phase 5: deliver all PDFs)
- `eisenstein-triple-product` (Step 10: deliver triple product PDF)

## Troubleshooting

If sending fails:
1. **SSL error / exit code 35**: Proxy not bypassed. Ensure `NO_PROXY=discord.com` and `HTTPS_PROXY=""` are set
2. **401 Unauthorized**: Bot token may have been rotated. Check Nova's token in `/home/claude-user/.openclaw/openclaw.json`
3. **413 Request Entity Too Large**: Discord file limit is 25MB. Compress the PDF first
4. **Connection timeout**: Try increasing timeout from 15 to 30 seconds
