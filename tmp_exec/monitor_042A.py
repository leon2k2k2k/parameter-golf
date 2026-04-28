#!/usr/bin/env python3
"""Monitor 042A vs baseline, 041A. Auto-launches 042C on completion (042B skipped — same as 042A with 2593982). Posts to Discord each tick."""

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
POD = "root@216.243.220.226 -p 12309"

LOG_BASE  = "/workspace/runs/039-neg-slope-screen-on-1797-base/baseline/train.log"
LOG_041A  = "/workspace/runs/041-shrunk-loop-late-activation-screen/train.log"
LOG_042A  = "/workspace/runs/042A-slope-anneal-0707-to-05-screen/train.log"
LOG_042B  = "/workspace/runs/042B-slope-anneal-0707-to-05-gradual-screen/train.log"

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
    r = subprocess.run(f"{SSH} {POD} \"{cmd}\"", shell=True, capture_output=True, text=True, timeout=20)
    return r.stdout.strip().splitlines()


def parse_log(lines):
    data = {}
    for line in lines:
        m = STEP_RE.match(line.strip())
        if m:
            data[int(m.group(1))] = (float(m.group(2)), m.group(3), int(m.group(4)))
    return data


def get_status(log):
    lines = ssh_lines(f"grep -E 'slope_anneal|stopping_early|post-ema val_bpb|post-ema val_loss|final_val_bpb|diagnostic.*val_bpb' {log} | tail -5")
    stopping     = any('stopping_early' in l for l in lines)
    slope_switch = next((l for l in lines if 'slope_anneal' in l), None)
    prequant_ema = next((l for l in lines if 'post-ema val_bpb' in l or 'post-ema val_loss' in l), None)
    final_quant  = next((l for l in lines if 'final_val_bpb' in l), None)
    return stopping, slope_switch, prequant_ema, final_quant


def launch_042C():
    subprocess.run(
        f"{SSH} {POD} \"setsid bash /workspace/launch_042C.sh </dev/null >/tmp/launch_042C.out 2>&1 & disown; echo launched\"",
        shell=True, timeout=15)


def main():
    buf = io.StringIO()
    out = sys.stdout
    sys.stdout = buf

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    a_lines = ssh_lines(f"grep 'train_loss' {LOG_042A} | tail -8")
    a_data   = parse_log(a_lines)
    stopping, slope_switch, prequant_ema, final_quant = get_status(LOG_042A)
    pod_lines = subprocess.run("runpodctl pod list", shell=True, capture_output=True, text=True).stdout

    if not a_data:
        print(f"● 042A — {now} | no step data yet (still compiling)")
    else:
        last_step = max(a_data)
        last_loss, last_time, last_toks = a_data[last_step]
        m_step = re.search(r'step:(\d+)', slope_switch) if slope_switch else None
        slope_tag = f"switched@{m_step.group(1)}" if m_step else ("switching" if slope_switch else "pre-switch")
        print(f"● 042A — step {last_step} | loss: {last_loss:.4f} | {last_time} | {last_toks//1000}K tok/s | {slope_tag}")
        print()

        steps = sorted(a_data.keys())
        step_filter = "|".join(f"^{s}/" for s in steps)
        ref_base = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_BASE}"))
        ref_041a = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_041A}"))

        col_w = 10
        print(f"  {'Step':>6}  {'Baseline':>{col_w}}  {'041A':>{col_w}}  {'042A':>{col_w}}")
        print("  " + "-"*6 + "  " + ("-"*col_w + "  ")*3)
        for s in steps:
            b  = f"{ref_base[s][0]:.4f}" if s in ref_base else "    —   "
            a1 = f"{ref_041a[s][0]:.4f}" if s in ref_041a else "    —   "
            a2 = f"{a_data[s][0]:.4f}"
            print(f"  {s:>6}  {b:>{col_w}}  {a1:>{col_w}}  {a2:>{col_w}}")
        print()

        done = False
        if final_quant:
            m = re.search(r'([\d.]+)', final_quant)
            val = float(m.group(1)) if m else None
            print(f"FINAL QUANT: {final_quant.strip()}")
            if val:
                if val < 1.0641:
                    print(f"   WIN — {val:.5f} < 1.0641 (baseline 1.06514)")
                elif val < 1.0670:
                    print(f"   NOISE ZONE — {val:.5f} in [1.0641, 1.0670]")
                else:
                    print(f"   KILL — {val:.5f} >= 1.0670")
            done = True
        elif prequant_ema:
            m = re.search(r'val_bpb[: ]+([\d.]+)', prequant_ema)
            val = float(m.group(1)) if m else None
            print(f"PRE-QUANT EMA: {prequant_ema.strip()}")
            if val:
                if val < 1.0641:
                    print(f"   WIN — {val:.5f} < 1.0641")
                elif val < 1.0670:
                    print(f"   NOISE ZONE — {val:.5f} in [1.0641, 1.0670]")
                else:
                    print(f"   KILL — {val:.5f} >= 1.0670")
            print("   GPTQ running...")
        elif stopping:
            print("stopping_early — GPTQ running...")

        if done:
            print("\n→ Launching 042C now (042B skipped)...")
            launch_042C()

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
