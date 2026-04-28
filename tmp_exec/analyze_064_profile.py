#!/usr/bin/env python3
"""Spec 064 — post-run analysis of the nsys profile.

Inputs:
  --sqlite        path to profile_full.sqlite (exported by `nsys export`)
  --train-log     path to train.log produced by the run
  --out-dir       directory to write summary files into
  --enable-looping-frac  ENABLE_LOOPING_AT value used (e.g. 0.35)
  --total-iterations     ITERATIONS env var used (e.g. 250)

Outputs (written to --out-dir):
  kernel_summary_pre_loop.txt   — top kernels by time, pre loop activation
  kernel_summary_post_loop.txt  — top kernels by time, post loop activation
  vram_curve.csv                — step, allocated_MiB, max_allocated_MiB
  toks_per_s.csv                — step, tok_per_s, wallclock_s

The pre/post-loop split is based on the loop activation step
(enable_looping_frac * total_iterations). We map step number to nsys
timeline time via tok/s entries scraped from train.log (each TRAIN_LOG_EVERY
step prints an elapsed time).

Robust to either form of nsys CUPTI kernel table (CUPTI_ACTIVITY_KIND_KERNEL
or the older CUPTI_KERNEL); falls back to whichever one exists.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
from pathlib import Path


def parse_train_log(path: Path):
    """Scrape (step, wallclock_s, tok_per_s, alloc_mib, max_alloc_mib) tuples
    from the training log. Tolerant of variations: returns whatever fields
    we can find, with None for missing.
    """
    rows = []
    step_re = re.compile(r"step\s*[=:\s]\s*(\d+)", re.IGNORECASE)
    toks_re = re.compile(r"(\d+(?:\.\d+)?)\s*(?:toks?/s|tokens?/s)", re.IGNORECASE)
    wall_re = re.compile(r"(?:elapsed|wallclock|wall)\s*[=:\s]\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
    alloc_re = re.compile(r"(?:mem|vram|alloc)[\w_]*\s*[=:\s]\s*(\d+(?:\.\d+)?)\s*(?:MiB|MB)", re.IGNORECASE)
    max_alloc_re = re.compile(r"max[\w_]*(?:mem|alloc)[\w_]*\s*[=:\s]\s*(\d+(?:\.\d+)?)\s*(?:MiB|MB)", re.IGNORECASE)
    for line in path.read_text(errors="replace").splitlines():
        m_step = step_re.search(line)
        if not m_step:
            continue
        step = int(m_step.group(1))
        m_toks = toks_re.search(line)
        m_wall = wall_re.search(line)
        m_alloc = alloc_re.search(line)
        m_max_alloc = max_alloc_re.search(line)
        rows.append({
            "step": step,
            "wallclock_s": float(m_wall.group(1)) if m_wall else None,
            "tok_per_s": float(m_toks.group(1)) if m_toks else None,
            "alloc_mib": float(m_alloc.group(1)) if m_alloc else None,
            "max_alloc_mib": float(m_max_alloc.group(1)) if m_max_alloc else None,
        })
    return rows


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})


def find_kernel_table(conn: sqlite3.Connection):
    """Return (table_name, time_col, name_id_col) for whichever kernel table
    nsys produced. Newer nsys: CUPTI_ACTIVITY_KIND_KERNEL with start, end,
    shortName columns. Older: CUPTI_KERNEL or KERNEL.
    """
    cur = conn.cursor()
    candidates = [
        ("CUPTI_ACTIVITY_KIND_KERNEL", "shortName"),
        ("CUPTI_KERNEL", "shortName"),
        ("KERNEL", "shortName"),
    ]
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    for tbl, name_col in candidates:
        if tbl in tables:
            return tbl, name_col
    return None, None


def kernel_ranking(conn: sqlite3.Connection, t_start_ns: int | None, t_end_ns: int | None, top_n: int = 25):
    tbl, name_id_col = find_kernel_table(conn)
    if tbl is None:
        return None, "No kernel table found in nsys sqlite (nothing to rank)."
    where = []
    params = []
    if t_start_ns is not None:
        where.append("k.start >= ?")
        params.append(t_start_ns)
    if t_end_ns is not None:
        where.append("k.end <= ?")
        params.append(t_end_ns)
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT s.value AS kernel_name,
               COUNT(*) AS launches,
               SUM(k.end - k.start) AS total_ns,
               AVG(k.end - k.start) AS avg_ns
        FROM {tbl} k
        JOIN StringIds s ON s.id = k.{name_id_col}
        {where_clause}
        GROUP BY s.value
        ORDER BY total_ns DESC
        LIMIT ?
    """
    params.append(top_n)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return None, f"No kernels found in window [{t_start_ns}, {t_end_ns}] of {tbl}."
    return rows, None


def format_ranking(rows, total_window_ns: int | None) -> str:
    out = []
    out.append(f"{'rank':>4}  {'kernel':<60}  {'launches':>10}  {'total_ms':>12}  {'avg_us':>10}  {'%win':>6}")
    out.append("-" * 110)
    sum_ns = sum(r[2] for r in rows) if rows else 0
    denom = total_window_ns if total_window_ns else sum_ns
    for i, (name, launches, total_ns, avg_ns) in enumerate(rows, 1):
        pct = (100.0 * total_ns / denom) if denom else 0.0
        out.append(f"{i:>4}  {name[:60]:<60}  {launches:>10}  {total_ns/1e6:>12.3f}  {avg_ns/1e3:>10.2f}  {pct:>6.2f}")
    return "\n".join(out)


def step_to_wallclock(rows, target_step):
    """Linearly interpolate wallclock_s for a given step from the parsed log."""
    pts = [(r["step"], r["wallclock_s"]) for r in rows if r["wallclock_s"] is not None]
    if not pts:
        return None
    pts.sort()
    if target_step <= pts[0][0]:
        return pts[0][1]
    if target_step >= pts[-1][0]:
        return pts[-1][1]
    for (s0, w0), (s1, w1) in zip(pts, pts[1:]):
        if s0 <= target_step <= s1:
            if s1 == s0:
                return w0
            f = (target_step - s0) / (s1 - s0)
            return w0 + f * (w1 - w0)
    return pts[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--enable-looping-frac", type=float, default=0.35)
    ap.add_argument("--total-iterations", type=int, default=250)
    args = ap.parse_args()

    sqlite_path = Path(args.sqlite)
    train_log = Path(args.train_log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sqlite_path.exists():
        print(f"[analyze_064] sqlite not found: {sqlite_path}", file=sys.stderr)
        sys.exit(1)
    if not train_log.exists():
        print(f"[analyze_064] train.log not found: {train_log}", file=sys.stderr)
        sys.exit(1)

    # ── 1. Scrape train.log for tok/s + VRAM curves ──────────────────────
    log_rows = parse_train_log(train_log)
    write_csv(out_dir / "toks_per_s.csv", log_rows,
              fields=["step", "wallclock_s", "tok_per_s"])
    write_csv(out_dir / "vram_curve.csv", log_rows,
              fields=["step", "alloc_mib", "max_alloc_mib"])

    # ── 2. Determine pre/post-loop windows ───────────────────────────────
    loop_step = int(args.enable_looping_frac * args.total_iterations)
    t_loop_s = step_to_wallclock(log_rows, loop_step)

    if t_loop_s is None:
        print("[analyze_064] could not infer loop activation wallclock from train.log; "
              "falling back to whole-run kernel ranking only", file=sys.stderr)
        windows = [("full_run", None, None)]
    else:
        # Window: 5 steps wide on each side (in step terms), translated to ns.
        t_5_steps_pre = step_to_wallclock(log_rows, max(loop_step - 5, 0))
        t_5_steps_post_start = step_to_wallclock(log_rows, loop_step + 1)
        t_5_steps_post_end = step_to_wallclock(log_rows, loop_step + 6)
        # nsys timeline starts at process start (~0). Wallclock from train.log
        # is also relative to start, so 1:1 mapping in seconds → ns.
        windows = [
            ("pre_loop", int((t_5_steps_pre or 0) * 1e9), int(t_loop_s * 1e9)),
            ("post_loop", int((t_5_steps_post_start or t_loop_s) * 1e9),
                          int((t_5_steps_post_end or t_loop_s + 5) * 1e9)),
        ]

    # ── 3. Query nsys sqlite for kernel rankings ─────────────────────────
    conn = sqlite3.connect(str(sqlite_path))
    try:
        for label, t0, t1 in windows:
            rows, err = kernel_ranking(conn, t0, t1)
            target = out_dir / f"kernel_summary_{label}.txt"
            if rows is None:
                target.write_text(f"[analyze_064] {err}\n")
                print(f"[analyze_064] {label}: {err}", file=sys.stderr)
                continue
            window_ns = (t1 - t0) if (t0 is not None and t1 is not None) else None
            header = (
                f"# kernel ranking — window={label}  "
                f"t0={t0}  t1={t1}  width_ms={(window_ns or 0)/1e6:.1f}\n"
                f"# loop_step={loop_step}  loop_wallclock_s={t_loop_s}\n"
                f"# from {sqlite_path.name}\n\n"
            )
            target.write_text(header + format_ranking(rows, window_ns) + "\n")
            print(f"[analyze_064] {label}: top kernel = {rows[0][0]} "
                  f"({rows[0][2]/1e6:.1f} ms over {rows[0][1]} launches)")
    finally:
        conn.close()

    print(f"[analyze_064] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
