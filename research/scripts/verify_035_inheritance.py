#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


IGNORE_KEYS = {
    "artifact_dir",
    "logfile",
    "model_path",
    "quantized_model_path",
    "run_id",
    "distributed",
    "world_size",
    "rank",
    "local_rank",
    "is_main_process",
    "grad_accum_steps",
}


def dump_hparams(record_dir: Path, env: dict[str, str]) -> dict[str, object]:
    code = r"""
import json
import os
import sys
sys.path.insert(0, os.getcwd())
import train_gpt
h = train_gpt.Hyperparameters()
current = {k: getattr(h, k) for k, _ in vars(type(h)).items() if not k.startswith("_")}
print(json.dumps(current, sort_keys=True, default=str))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=record_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--expected-min-lr", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    record_dir = Path(args.record_dir)
    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_env = os.environ.copy()
    baseline_env = current_env.copy()
    baseline_env["MIN_LR"] = "0.0"

    baseline = dump_hparams(record_dir, baseline_env)
    current = dump_hparams(record_dir, current_env)

    (out_dir / "config.json").write_text(
        json.dumps(current, indent=2, sort_keys=True, default=str) + "\n"
    )

    expected_min_lr = str(float(args.expected_min_lr))
    diffs: dict[str, dict[str, str]] = {}
    for key, baseline_value in baseline.items():
        if key in IGNORE_KEYS:
            continue
        current_value = str(current.get(key))
        if key == "min_lr":
            if current_value != expected_min_lr:
                diffs[key] = {"baseline": str(baseline_value), "current": current_value}
            continue
        if current_value != str(baseline_value):
            diffs[key] = {"baseline": str(baseline_value), "current": current_value}

    (out_dir / "config_diff.json").write_text(
        json.dumps(diffs, indent=2, sort_keys=True) + "\n"
    )
    if diffs:
        raise SystemExit(
            f"Invalid {args.label} config drift: " + json.dumps(diffs, sort_keys=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
