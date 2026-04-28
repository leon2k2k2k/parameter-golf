#!/usr/bin/env python3
"""Monitor 041H vs baseline, 041A. Posts to Discord each tick."""

import subprocess, re, json, os, io, sys
from datetime import datetime, timezone
import requests

os.environ['NO_PROXY'] = 'discord.com'
os.environ['HTTPS_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''

DISCORD_TOKEN   = os.environ["DISCORD_BOT_TOKEN"]  # set DISCORD_BOT_TOKEN before running
DISCORD_CHANNEL = "1474618189806833745"
DISCORD_URL     = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages"

SSH = "ssh -i /home/claude-user/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
POD = "root@216.243.220.218 -p 15498"

LOG_BASE = "/workspace/runs/039-neg-slope-screen-on-1797-base/baseline/train.log"
LOG_041A = "/workspace/runs/041-shrunk-loop-late-activation-screen/train.log"
LOG_041H = "/workspace/runs/041H-loop45-frac035-screen/train.log"

STEP_RE = re.compile(r'^(\d+)/\d+ train_loss:\s*([\d.]+) train_time:\s*([\S]+) tok/s:\s*(\d+)')


def discord_post(msg):
    try:
        requests.post(DISCORD_URL,
            headers={"Authorization": f"Bot {DISCORD_TOKEN}", "Content-Type": "application/json"},
            json={"content": f"From Claude Code:\n```\n{msg[:1900]}\n```"},
            timeout=15)
    except Exception:
        pass


def ssh_lines(cmd):
    r = subprocess.run(f"{SSH} {POD} \"{cmd}\"", shell=True, capture_output=True, text=True, timeout=15)
    return r.stdout.strip().splitlines()


def parse_log(lines):
    data = {}
    for line in lines:
        m = STEP_RE.match(line.strip())
        if m:
            data[int(m.group(1))] = (float(m.group(2)), m.group(3), int(m.group(4)))
    return data


def get_status(log):
    lines = ssh_lines(f"grep -E 'layer_loop|stopping_early|final_ema' {log} | tail -3")
    activated = any('layer_loop:enabled' in l for l in lines)
    stopping  = any('stopping_early' in l for l in lines)
    final_ema = next((l for l in lines if 'final_ema' in l), None)
    loop_step = None
    for l in lines:
        m = re.search(r'layer_loop:enabled step:(\d+)', l)
        if m: loop_step = int(m.group(1))
    return activated, stopping, final_ema, loop_step


def main():
    buf = io.StringIO()
    out = sys.stdout
    sys.stdout = buf

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    h_lines = ssh_lines(f"grep 'train_loss' {LOG_041H} | tail -8")
    h_data   = parse_log(h_lines)
    activated, stopping, final_ema, loop_step = get_status(LOG_041H)
    pod_lines = subprocess.run("runpodctl pod list", shell=True, capture_output=True, text=True).stdout

    if not h_data:
        print(f"● 041H — {now} | no step data yet (still compiling)")
    else:
        last_step = max(h_data)
        last_loss, last_time, last_toks = h_data[last_step]
        loop_tag = f"loop@{loop_step}" if activated else "no-loop"
        print(f"● 041H — step {last_step} | loss: {last_loss:.4f} | {last_time} | {last_toks//1000}K tok/s | {loop_tag}")
        print()

        steps = sorted(h_data.keys())
        step_filter = "|".join(f"^{s}/" for s in steps)
        ref_base = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_BASE}"))
        ref_041a = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_041A}"))

        col_w = 10
        print(f"  {'Step':>6}  {'Baseline':>{col_w}}  {'041A':>{col_w}}  {'041H':>{col_w}}")
        print("  " + "-"*6 + "  " + ("-"*col_w + "  ")*3)
        for s in steps:
            b = f"{ref_base[s][0]:.4f}" if s in ref_base else "    —   "
            a = f"{ref_041a[s][0]:.4f}" if s in ref_041a else "    —   "
            h = f"{h_data[s][0]:.4f}"
            print(f"  {s:>6}  {b:>{col_w}}  {a:>{col_w}}  {h:>{col_w}}")
        print()

        if final_ema:
            m = re.search(r'val_bpb[:\s]*([\d.]+)', final_ema)
            val = float(m.group(1)) if m else None
            print(f"FINAL EMA: {final_ema.strip()}")
            if val:
                if val < 1.0641:
                    print(f"   WIN — {val:.5f} < 1.0641 (baseline 1.06514)")
                elif val < 1.0670:
                    print(f"   NOISE ZONE — {val:.5f} in [1.0641, 1.0670] -> run second seed")
                else:
                    print(f"   KILL — {val:.5f} >= 1.0670")
            print("   STOP POD cxioitzx6s2v7k NOW")
        elif stopping:
            print("stopping_early detected — check log")

    try:
        pods = json.loads(pod_lines)
        print(f"\n  {'Pod':<20} {'ID':<20} {'GPU':>5}  {'$/hr':>6}")
        for p in pods:
            print(f"  {p['name']:<20} {p['id']:<20} {p['gpuCount']:>5}x  {p['costPerHr']:>6.2f}")
    except Exception:
        print(pod_lines)

    sys.stdout = out
    output = buf.getvalue()
    print(output, end="")
    discord_post(output)


if __name__ == "__main__":
    main()
