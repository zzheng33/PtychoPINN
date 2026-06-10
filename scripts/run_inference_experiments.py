#!/usr/bin/env python
"""Run PtychoPINN inference sweeps while logging GPU power."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR_SCRIPT = REPO_ROOT / "scripts" / "monitor_gpu_power.py"
INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "run_inference.py"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def short_gpu_name(name: str) -> str:
    normalized = name.upper().replace("_", " ").replace("-", " ")
    known_names = (
        "A100",
        "H100",
        "H200",
        "B200",
        "MI300X",
        "MI300A",
        "MAX",
    )
    for known_name in known_names:
        if known_name in normalized:
            return "Max" if known_name == "MAX" else known_name
    return safe_name(name)


def detect_gpu_label(args) -> tuple[str, str]:
    if args.gpu_label:
        vendor = args.vendor if args.vendor != "auto" else "unknown_vendor"
        return vendor, safe_name(args.gpu_label)

    cmd = [
        sys.executable,
        str(MONITOR_SCRIPT),
        "--vendor",
        args.vendor,
        "--list-gpus",
    ]
    if args.devices:
        cmd.extend(["--devices", args.devices])

    result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    rows = [line.split(",", 2) for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return args.vendor, "unknown_gpu"

    vendor = rows[0][0]
    names = [row[2] for row in rows if len(row) == 3]
    unique_names = sorted(set(short_gpu_name(name) for name in names))
    if len(unique_names) == 1:
        return vendor, unique_names[0]
    return vendor, safe_name("_".join(unique_names))


def run_one(args, dataset: str, batch_size: int, run_dir: Path) -> dict[str, object]:
    label = f"{dataset}_bs{batch_size}"
    dataset_dir = run_dir / safe_name(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    power_csv = dataset_dir / f"bs{batch_size}_power.csv"
    output_dir = dataset_dir / f"bs{batch_size}_outputs"

    monitor_cmd = [
        sys.executable,
        str(MONITOR_SCRIPT),
        "--vendor",
        args.vendor,
        "--interval",
        str(args.interval),
        "--output",
        str(power_csv),
        "--label",
        label,
    ]
    if args.devices:
        monitor_cmd.extend(["--devices", args.devices])

    inference_cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--dataset",
        dataset,
        "--device",
        args.device,
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
    ]
    if args.model_key:
        inference_cmd.extend(["--model-key", args.model_key])
    if args.run_id:
        inference_cmd.extend(["--run-id", args.run_id])
    if args.config:
        inference_cmd.extend(["--config", args.config])

    print(f"Starting monitor: {power_csv}", flush=True)
    monitor = subprocess.Popen(monitor_cmd, cwd=REPO_ROOT)
    time.sleep(args.warmup_seconds)

    start = time.time()
    completed = None
    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    try:
        print(f"Running inference: dataset={dataset}, batch_size={batch_size}", flush=True)
        completed = subprocess.run(inference_cmd, cwd=REPO_ROOT, env=env)
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()

    end = time.time()
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "device": args.device,
        "vendor": args.vendor,
        "detected_vendor": args.detected_vendor,
        "gpu_label": args.gpu_label,
        "devices": args.devices or "all",
        "returncode": completed.returncode if completed is not None else 1,
        "duration_s": f"{end - start:.6f}",
        "power_csv": str(power_csv),
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["TP1", "TP2", "IC1", "IC2", "NCM", "FLY1", "LFP", "W", "LCLS"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1,2,4,8,16,32,64,128,256,512,1024] )
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--vendor", default="auto", choices=("auto", "nvidia", "amd", "intel"))
    parser.add_argument("--devices", default=None, help="Comma-separated GPU indices to monitor.")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--conda-env", default="ptychopinn_torch", help="Kept for wrapper compatibility.")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("power_experiments"))
    parser.add_argument("--gpu-label", default=None, help="Manual output-folder GPU label, e.g. A100 or MI300X.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    detected_vendor, gpu_label = detect_gpu_label(args)
    args.detected_vendor = detected_vendor
    args.gpu_label = gpu_label
    run_dir = args.output_root / gpu_label
    run_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            row = run_one(args, dataset, batch_size, run_dir)
            if row["returncode"] != 0 and not args.continue_on_error:
                print(f"Stopping after failed run: {row}", file=sys.stderr)
                return int(row["returncode"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
