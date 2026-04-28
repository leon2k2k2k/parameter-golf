#!/usr/bin/env python3
"""Monitor 042C vs baseline, 041A, 042A. On completion: rsync logs, stop pod, post Discord report."""

import subprocess, re, json, os, io, sys
from datetime import datetime, timezone
import requests

os.environ['NO_PROXY'] = 'discord.com'
os.environ['HTTPS_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''

DISCORD_TOKEN   = os.environ["DISCORD_BOT_TOKEN"]  # set DISCORD_BOT_TOKEN before running
DISCORD_CHANNEL = "1474618189806833745"
DISCORD_URL     = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages"

SSH     = "ssh -i /home/claude-user/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
RSYNC   = "rsync -av -e 'ssh -i /home/claude-user/.runpod/ssh/RunPod-Key-Go -p 12309 -o StrictHostKeyChecking=no'"
POD     = "root@216.243.220.226 -p 12309"
POD_ID  = "9cswcusasowrdd"

LOG_BASE  = "/workspace/runs/039-neg-slope-screen-on-1797-base/baseline/train.log"
LOG_041A  = "/workspace/runs/041-shrunk-loop-late-activation-screen/train.log"
LOG_042A  = "/workspace/runs/042A-slope-anneal-0707-to-05-screen/train.log"
LOG_042C  = "/workspace/runs/042C-slope-anneal-0707-to-0-screen/train.log"

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
    lines = ssh_lines(f"grep -E 'slope_anneal|stopping_early|post-ema val_bpb|final_val_bpb' {log} | tail -5")
    stopping     = any('stopping_early' in l for l in lines)
    slope_switch = next((l for l in lines if 'slope_anneal' in l), None)
    prequant_ema = next((l for l in lines if 'post-ema val_bpb' in l), None)
    final_quant  = next((l for l in lines if 'final_val_bpb' in l), None)
    return stopping, slope_switch, prequant_ema, final_quant


def get_final_val(log_remote, label):
    lines = ssh_lines(f"grep -E 'post-ema val_bpb|final_val_bpb' {log_remote} | tail -3")
    result = []
    for l in lines:
        if 'final_val_bpb' in l:
            m = re.search(r'([\d.]+)', l)
            if m:
                result.append(f"{label} final_val_bpb: {m.group(1)}")
        elif 'post-ema val_bpb' in l:
            m = re.search(r'val_bpb:([\d.]+)', l)
            if m:
                result.append(f"{label} pre-quant EMA: {m.group(1)}")
    return result


def rsync_and_stop():
    runs_local = "/home/claude-user/ai-workspace/projects/parameter-golf/runs"
    for slug in ["042A-slope-anneal-0707-to-05-screen", "042C-slope-anneal-0707-to-0-screen"]:
        os.makedirs(f"{runs_local}/{slug}", exist_ok=True)
        subprocess.run(
            f"rsync -av -e 'ssh -i /home/claude-user/.runpod/ssh/RunPod-Key-Go -p 12309 -o StrictHostKeyChecking=no' "
            f"root@216.243.220.226:/workspace/runs/{slug}/ {runs_local}/{slug}/",
            shell=True, timeout=120)
    subprocess.run(f"runpodctl pod stop {POD_ID}", shell=True, timeout=30)


def judge(val):
    if val < 1.0641:
        return f"WIN — {val:.5f} < 1.0641"
    elif val < 1.0670:
        return f"NOISE ZONE — {val:.5f} in [1.0641, 1.0670]"
    else:
        return f"KILL — {val:.5f} >= 1.0670"


def main():
    buf = io.StringIO()
    out = sys.stdout
    sys.stdout = buf

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    c_lines = ssh_lines(f"grep 'train_loss' {LOG_042C} | tail -8")
    c_data   = parse_log(c_lines)
    stopping, slope_switch, prequant_ema, final_quant = get_status(LOG_042C)
    pod_lines = subprocess.run("runpodctl pod list", shell=True, capture_output=True, text=True).stdout

    if not c_data:
        print(f"● 042C — {now} | no step data yet (still compiling)")
    else:
        last_step = max(c_data)
        last_loss, last_time, last_toks = c_data[last_step]
        slope_tag = "switched" if slope_switch else "pre-switch"
        print(f"● 042C — step {last_step} | loss: {last_loss:.4f} | {last_time} | {last_toks//1000}K tok/s | {slope_tag}")
        print()

        steps = sorted(c_data.keys())
        step_filter = "|".join(f"^{s}/" for s in steps)
        ref_base = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_BASE}"))
        ref_041a = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_041A}"))
        ref_042a = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_042A}"))

        col_w = 10
        print(f"  {'Step':>6}  {'Baseline':>{col_w}}  {'041A':>{col_w}}  {'042A':>{col_w}}  {'042C':>{col_w}}")
        print("  " + "-"*6 + "  " + ("-"*col_w + "  ")*4)
        for s in steps:
            b  = f"{ref_base[s][0]:.4f}" if s in ref_base else "    —   "
            a1 = f"{ref_041a[s][0]:.4f}" if s in ref_041a else "    —   "
            a2 = f"{ref_042a[s][0]:.4f}" if s in ref_042a else "    —   "
            c2 = f"{c_data[s][0]:.4f}"
            print(f"  {s:>6}  {b:>{col_w}}  {a1:>{col_w}}  {a2:>{col_w}}  {c2:>{col_w}}")
        print()

        done = False
        if final_quant:
            m = re.search(r'([\d.]+)', final_quant)
            val = float(m.group(1)) if m else None
            print(f"042C FINAL QUANT: {final_quant.strip()}")
            if val:
                print(f"   {judge(val)}")
            done = True
        elif prequant_ema:
            m = re.search(r'val_bpb:([\d.]+)', prequant_ema)
            val = float(m.group(1)) if m else None
            print(f"042C PRE-QUANT EMA: {prequant_ema.strip()}")
            if val:
                print(f"   {judge(val)}")
            print("   GPTQ running...")
        elif stopping:
            print("042C stopping_early — GPTQ running...")

        if done:
            print("\n=== FINAL REPORT ===")
            for line in get_final_val(LOG_042A, "042A"):
                print(f"  {line}")
            for line in get_final_val(LOG_042C, "042C"):
                print(f"  {line}")
            print("  Baseline pre-quant EMA: 1.06514")
            print("\n→ Rsyncing logs and stopping pod...")
            sys.stdout = out
            output = buf.getvalue()
            print(output, end="")
            discord_post(output)
            rsync_and_stop()
            subprocess.run("runpodctl pod list", shell=True)
            return

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
