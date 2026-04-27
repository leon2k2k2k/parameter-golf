#!/usr/bin/env python3
"""Monitor 051 PPM-D vs 050A baseline. Highlights throughput at pre/post loop activation."""

import subprocess, re, sys, io, os
from datetime import datetime, timezone
import requests

os.environ['NO_PROXY'] = 'discord.com'
os.environ['HTTPS_PROXY'] = ''
os.environ['HTTP_PROXY'] = ''

DISCORD_TOKEN   = "MTQ3NDQxMDYyODUyNDI4MTg3Nw.Gw6q3X.zZ4XLVCvsK57Wqht3U5dHQTpBedOpw9SfgObzw"
DISCORD_CHANNEL = "1474618189806833745"
DISCORD_URL     = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages"

SSH = "ssh -i /home/claude-user/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
POD = "FILL_IN_HOST -p FILL_IN_PORT"  # set before launch

LOG_050A = "/workspace/runs/050A-ac-fix-screen/train.log"
LOG_051  = "/workspace/runs/051-perpass-mlp-untied-screen/train.log"

STEP_RE = re.compile(r'^(\d+)/\d+ train_loss:\s*([\d.]+) train_time:\s*([\S]+) tok/s:\s*(\d+)')

# 050A reference throughput (tok/s) — fill after 050A screen completes
# Pre-loop: ~4300K, post-loop: ~3200K (rough; update with actuals)
REF_PRE_LOOP_TOKS  = None  # will be read from 050A log
REF_POST_LOOP_TOKS = None

# Loop activates at ENABLE_LOOPING_AT=0.35 fraction of 20min = ~7min = ~step 1750
LOOP_ACTIVATION_STEP = 1750


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


def interval_toks(data, step_a, step_b):
    """Compute tok/s between two steps using interval formula (not cumulative avg)."""
    if step_a not in data or step_b not in data:
        return None
    t_a = float(data[step_a][1].replace('s',''))
    t_b = float(data[step_b][1].replace('s',''))
    # tokens per step = TRAIN_BATCH_TOKENS = 786432
    tokens = (step_b - step_a) * 786432
    dt = t_b - t_a
    if dt <= 0:
        return None
    return int(tokens / dt)


def throughput_block(data_050a, data_051):
    """Show pre/post loop throughput for both arms using interval formula."""
    lines = []

    # Pre-loop: steps 100→400 (well before loop activation ~1750)
    pre_steps = sorted([s for s in data_051 if 100 <= s <= 400])
    if len(pre_steps) >= 2:
        s0, s1 = pre_steps[0], pre_steps[-1]
        toks_051 = interval_toks(data_051, s0, s1)
        toks_050a = interval_toks(data_050a, s0, s1) if s0 in data_050a and s1 in data_050a else None
        tag = ""
        if toks_051 and toks_050a:
            delta_pct = (toks_051 - toks_050a) / toks_050a * 100
            tag = f"  ({delta_pct:+.1f}% vs 050A)"
        lines.append(f"  Pre-loop  (steps {s0}→{s1}):  051={toks_051//1000 if toks_051 else '?'}K tok/s  |  050A={toks_050a//1000 if toks_050a else '?'}K tok/s{tag}")

    # Post-loop: steps after ~1800, use last two available checkpoints
    post_steps = sorted([s for s in data_051 if s > LOOP_ACTIVATION_STEP + 100])
    if len(post_steps) >= 2:
        s0, s1 = post_steps[0], post_steps[-1]
        toks_051 = interval_toks(data_051, s0, s1)
        toks_050a = interval_toks(data_050a, s0, s1) if s0 in data_050a and s1 in data_050a else None
        tag = ""
        if toks_051 and toks_050a:
            delta_pct = (toks_051 - toks_050a) / toks_050a * 100
            tag = f"  ({delta_pct:+.1f}% vs 050A)"
        lines.append(f"  Post-loop (steps {s0}→{s1}):  051={toks_051//1000 if toks_051 else '?'}K tok/s  |  050A={toks_050a//1000 if toks_050a else '?'}K tok/s{tag}")

    return lines


def main():
    buf = io.StringIO()
    sys.stdout = buf

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    lines_050a = ssh_lines(f"grep 'train_loss' {LOG_050A} | tail -30")
    lines_051  = ssh_lines(f"grep 'train_loss' {LOG_051}  | tail -30")
    data_050a  = parse_log(lines_050a)
    data_051   = parse_log(lines_051)

    # Final / prequant status
    final_lines = ssh_lines(f"grep -E 'stopping_early|post-ema val_bpb|diagnostic.*val_bpb|final_val_bpb|looping_active' {LOG_051} | tail -6")
    stopping    = any('stopping_early' in l for l in final_lines)
    prequant    = next((l for l in final_lines if 'post-ema val_bpb' in l or ('diagnostic' in l and 'val_bpb' in l)), None)
    final_quant = next((l for l in final_lines if 'final_val_bpb' in l), None)
    loop_on     = any('looping_active' in l for l in final_lines)

    pod_out = subprocess.run("runpodctl pod list", shell=True, capture_output=True, text=True).stdout

    print(f"● 051 PPM-D monitor — {now}")
    print(f"  Loop activation: {'YES' if loop_on else 'not yet'} (~step {LOOP_ACTIVATION_STEP})")
    print()

    # Loss table
    if data_051:
        last_step = max(data_051)
        last_loss, last_time, last_toks = data_051[last_step]
        print(f"  Last step: {last_step} | loss: {last_loss:.4f} | elapsed: {last_time} | tok/s: {last_toks//1000}K")
        print()

        steps = sorted(data_051.keys())
        step_filter = "|".join(f"^{s}/" for s in steps)
        ref = parse_log(ssh_lines(f"grep -E '{step_filter}' {LOG_050A}"))

        print(f"  {'Step':>6}  {'050A':>10}  {'051':>10}  {'Δ':>8}")
        print("  " + "-"*6 + "  " + "-"*10 + "  " + "-"*10 + "  " + "-"*8)
        for s in steps:
            r = f"{ref[s][0]:.4f}" if s in ref else "    —   "
            a = f"{data_051[s][0]:.4f}"
            if s in ref:
                delta = data_051[s][0] - ref[s][0]
                d = f"{delta:+.4f}"
            else:
                d = "    —"
            print(f"  {s:>6}  {r:>10}  {a:>10}  {d:>8}")
        print()

        # Throughput comparison
        tput_lines = throughput_block(ref, data_051)
        if tput_lines:
            print("  ── Throughput (interval formula) ──")
            for l in tput_lines:
                print(l)
            print()
    else:
        print("  No step data yet (compiling or not started)")
        print()

    # Final verdict
    if final_quant:
        m = re.search(r'([\d.]+)', final_quant)
        val = float(m.group(1)) if m else None
        print(f"  FINAL QUANT: {final_quant.strip()}")
        if val:
            delta = val - 1.07387  # 050A post-quant
            verdict = "WIN" if val < 1.073 else ("NOISE" if val < 1.076 else "KILL")
            print(f"  → {verdict}  Δ vs 050A quant: {delta:+.5f}")
    elif prequant:
        m = re.search(r'val_bpb[: ]+([\d.]+)', prequant)
        val = float(m.group(1)) if m else None
        print(f"  PRE-QUANT EMA: {prequant.strip()}")
        if val:
            delta = val - 1.06484  # 050A pre-quant
            verdict = "WIN" if val <= 1.064 else ("NOISE" if val <= 1.068 else "KILL")
            print(f"  → {verdict}  Δ vs 050A pre-quant: {delta:+.5f}")
        print("  GPTQ running...")
    elif stopping:
        print("  stopping_early detected — GPTQ running...")

    # Pod list
    print()
    print("  ── Pods ──")
    print(pod_out.strip())

    sys.stdout = sys.__stdout__
    output = buf.getvalue()
    print(output, end="")
    discord_post(output)


if __name__ == "__main__":
    main()
