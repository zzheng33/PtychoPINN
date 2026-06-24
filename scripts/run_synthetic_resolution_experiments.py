#!/usr/bin/env python
"""Run end-to-end resolution inference sweeps while logging GPU power."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR_SCRIPT = REPO_ROOT / "scripts" / "monitor_gpu_power.py"
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_resolution_inference_e2e.py"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def short_gpu_name(name: str) -> str:
    normalized = name.upper().replace("_", " ").replace("-", " ")
    for known_name in ("V100", "A100", "H100", "H200", "B200", "MI300X", "MI300A", "MAX"):
        if known_name in normalized:
            return "Max" if known_name == "MAX" else known_name
    return safe_name(name)


def detect_gpu_label(args) -> tuple[str, str]:
    if args.gpu_label:
        vendor = args.vendor if args.vendor != "auto" else "unknown_vendor"
        return vendor, safe_name(args.gpu_label)
    cmd = [sys.executable, str(MONITOR_SCRIPT), "--vendor", args.vendor, "--list-gpus"]
    if args.devices:
        cmd.extend(["--devices", args.devices])
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    rows = [line.split(",", 2) for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return args.vendor, "unknown_gpu"
    vendor = rows[0][0]
    names = [row[2] for row in rows if len(row) == 3]
    unique_names = sorted(set(short_gpu_name(name) for name in names))
    return vendor, unique_names[0] if len(unique_names) == 1 else safe_name("_".join(unique_names))


def run_one(args, resolution: int, batch_size: int, run_dir: Path) -> int:
    label = "N{}_bs{}".format(resolution, batch_size)
    output_dir = run_dir / "N{}".format(resolution)
    output_dir.mkdir(parents=True, exist_ok=True)
    power_csv = output_dir / "bs{}_power.csv".format(batch_size)
    timing_csv = output_dir / "bs{}_timing.csv".format(batch_size)

    prepared_root = args.prepared_root if args.prepared_root.is_absolute() else REPO_ROOT / args.prepared_root
    memmap_root = args.memmap_root if args.memmap_root.is_absolute() else REPO_ROOT / args.memmap_root
    memmap_dir = memmap_root / safe_name(args.dataset) / "N{}".format(resolution) / "bs{}".format(batch_size)

    prepare_cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--dataset",
        args.dataset,
        "--resolution",
        str(resolution),
        "--prepared-root",
        str(prepared_root),
        "--prepare-only",
    ]
    if args.remake_data:
        prepare_cmd.append("--remake-data")

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

    benchmark_cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--dataset",
        args.dataset,
        "--resolution",
        str(resolution),
        "--batch-size",
        str(batch_size),
        "--channels",
        str(args.channels),
        "--device",
        args.device,
        "--prepared-root",
        str(prepared_root),
        "--memmap-dir",
        str(memmap_dir),
        "--csv",
        str(timing_csv),
    ]
    if args.amp:
        benchmark_cmd.append("--amp")
    if args.remake_map:
        benchmark_cmd.append("--remake-map")

    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices
        env["HIP_VISIBLE_DEVICES"] = args.devices
        env["ROCR_VISIBLE_DEVICES"] = args.devices
        env["GPU_DEVICE_ORDINAL"] = args.devices
        env["ZE_AFFINITY_MASK"] = args.devices

    print("Preparing resized dataset outside power measurement: N={}".format(resolution), flush=True)
    subprocess.run(prepare_cmd, cwd=REPO_ROOT, env=env, check=True)

    monitor = None
    if args.test:
        print("Test mode: skipping GPU power monitor for {}.".format(label), flush=True)
    else:
        print("Starting monitor: {}".format(power_csv), flush=True)
        monitor_env = env.copy()
        monitor_env.pop("ZE_AFFINITY_MASK", None)
        monitor = subprocess.Popen(monitor_cmd, cwd=REPO_ROOT, env=monitor_env)
        time.sleep(args.warmup_seconds)

    try:
        print("Running end-to-end benchmark: dataset={}, N={}, batch_size={}".format(args.dataset, resolution, batch_size), flush=True)
        completed = subprocess.run(benchmark_cmd, cwd=REPO_ROOT, env=env)
        return int(completed.returncode)
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TP1", help="Dataset key or directory path to resize and replay.")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--vendor", choices=("auto", "nvidia", "amd", "intel"), default="auto")
    parser.add_argument("--devices", default=None, help="Comma-separated GPU indices to monitor/use.")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=Path("power_experiments_synthetic_resolution"))
    parser.add_argument("--prepared-root", type=Path, default=Path("synthetic_resolution_data"))
    parser.add_argument(
        "--memmap-root",
        type=Path,
        default=Path(os.environ["MEMMAP_ROOT"]) if os.environ.get("MEMMAP_ROOT") else Path("synthetic_resolution_memmaps"),
    )
    parser.add_argument("--gpu-label", default=None)
    parser.add_argument("--remake-data", action="store_true")
    parser.add_argument("--remake-map", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run without starting the GPU power monitor.")
    args = parser.parse_args()

    _vendor, gpu_label = (args.vendor, args.gpu_label or "test") if args.test else detect_gpu_label(args)
    run_dir = args.output_root / gpu_label
    run_dir.mkdir(parents=True, exist_ok=True)

    for resolution in args.resolutions:
        for batch_size in args.batch_sizes:
            returncode = run_one(args, resolution, batch_size, run_dir)
            if returncode != 0 and not args.continue_on_error:
                return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
