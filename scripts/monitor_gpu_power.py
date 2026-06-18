#!/usr/bin/env python
"""Sample GPU power, memory, and utilization to CSV."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_devices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def clean_number(value: str) -> float:
    value = value.strip()
    if value in {"", "N/A", "[Not Supported]"}:
        return math.nan
    return float(value)


def power_to_watts(value) -> float:
    if value in (None, "N/A"):
        return math.nan
    power = float(value)
    if power >= 1_000_000:
        return power / 1_000_000
    if power >= 1_000:
        return power / 1_000
    return power


def memory_to_mib(value) -> float:
    if value in (None, "N/A"):
        return math.nan
    mem = float(value)
    if mem > 1024 * 1024:
        return mem / (1024 * 1024)
    if mem > 1024:
        return mem / 1024
    return mem


def find_sysfs_power_files() -> list[str]:
    patterns = (
        "/sys/class/drm/card*/device/hwmon/hwmon*/power1_input",
        "/sys/class/drm/card*/device/hwmon/hwmon*/power1_average",
    )
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def find_intel_sysfs_power_files() -> list[tuple[int, str, str]]:
    files = []
    for card_path in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            vendor_path = Path(card_path) / "vendor"
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor != "0x8086":
            continue

        card_name = Path(card_path).parent.name
        try:
            gpu_index = int(card_name.replace("card", ""))
        except ValueError:
            gpu_index = len(files)

        for pattern in ("hwmon/hwmon*/power1_input", "hwmon/hwmon*/power1_average"):
            matches = sorted(glob.glob(str(Path(card_path) / pattern)))
            if matches:
                files.append((gpu_index, card_name, matches[0]))
                break
    return files


def read_sysfs_power_watts(power_files: list[str], gpu_index: int) -> float:
    if gpu_index >= len(power_files):
        return math.nan
    try:
        with open(power_files[gpu_index], "r", encoding="utf-8") as handle:
            return float(handle.read().strip()) / 1_000_000
    except OSError:
        return math.nan


def read_named_sysfs_power_watts(power_file: str) -> float:
    try:
        with open(power_file, "r", encoding="utf-8") as handle:
            return float(handle.read().strip()) / 1_000_000
    except OSError:
        return math.nan


def first_number(value: object) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not match:
        return math.nan
    return float(match.group(0))


def run_text(cmd: list[str], timeout: float = 5.0, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, env=env)
    return result.stdout


def geopmread_cmd(*args: object) -> list[str]:
    geopmread = shutil.which("geopmread")
    if geopmread is None:
        raise RuntimeError("geopmread was not found")
    return [geopmread, *(str(arg) for arg in args)]


def geopmread_env() -> dict[str, str]:
    env = os.environ.copy()
    # Aurora's /usr/bin/geopmread uses a /usr/bin/python3 shebang. If the
    # PtychoPINN venv or Python module paths leak into that interpreter, it can
    # import incompatible pkg_resources. Keep system paths for GEOPM libraries,
    # but remove Python-specific overrides.
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "VIRTUAL_ENV"):
        env.pop(name, None)
    return env


class NvidiaSampler:
    vendor = "nvidia"

    def __init__(self, devices: list[int] | None):
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi was not found")
        self.devices = devices

    def sample(self) -> list[dict[str, object]]:
        query = "index,name,power.draw,memory.used,utilization.gpu"
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        if self.devices is not None:
            cmd.insert(1, "-i")
            cmd.insert(2, ",".join(str(device) for device in self.devices))

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        rows = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index, name, power, mem_used, util = [part.strip() for part in line.split(",", 4)]
            rows.append(
                {
                    "vendor": self.vendor,
                    "gpu_index": int(index),
                    "gpu_name": name,
                    "power_w": clean_number(power),
                    "memory_used_mib": clean_number(mem_used),
                    "utilization_pct": clean_number(util),
                }
            )
        return rows


class AmdSampler:
    vendor = "amd"

    def __init__(self, devices: list[int] | None):
        self.devices = devices
        self.sysfs_power_files = find_sysfs_power_files()
        self.amdsmi = None
        self.handles = []
        try:
            import amdsmi  # type: ignore

            self.amdsmi = amdsmi
            self.amdsmi.amdsmi_init()
            self.handles = list(self.amdsmi.amdsmi_get_processor_handles())
        except Exception:
            self.amdsmi = None

        if self.devices is None:
            if self.handles:
                self.devices = list(range(len(self.handles)))
            else:
                self.devices = list(range(len(self.sysfs_power_files)))

        if not self.devices:
            raise RuntimeError("No AMD GPUs or sysfs power files were found")

    def close(self) -> None:
        if self.amdsmi is not None:
            try:
                self.amdsmi.amdsmi_shut_down()
            except Exception:
                pass

    def _amdsmi_value(self, call, default):
        try:
            return call()
        except Exception:
            return default

    def sample(self) -> list[dict[str, object]]:
        rows = []
        for gpu_index in self.devices or []:
            name = f"amd:{gpu_index}"
            power_w = read_sysfs_power_watts(self.sysfs_power_files, gpu_index)
            memory_used_mib = math.nan
            utilization_pct = math.nan

            if self.amdsmi is not None and gpu_index < len(self.handles):
                handle = self.handles[gpu_index]
                power_info = self._amdsmi_value(lambda: self.amdsmi.amdsmi_get_power_info(handle), {})
                for key in ("current_socket_power", "average_socket_power", "socket_power"):
                    value = power_to_watts(power_info.get(key))
                    if not math.isnan(value) and value > 0:
                        power_w = value
                        break

                vram_info = self._amdsmi_value(lambda: self.amdsmi.amdsmi_get_gpu_vram_usage(handle), {})
                memory_used_mib = memory_to_mib(vram_info.get("vram_used"))

                activity = self._amdsmi_value(lambda: self.amdsmi.amdsmi_get_gpu_activity(handle), {})
                utilization_pct = clean_number(str(activity.get("gfx_activity", "N/A")))

                info = self._amdsmi_value(lambda: self.amdsmi.amdsmi_get_gpu_asic_info(handle), {})
                name = str(info.get("market_name") or info.get("vendor_id") or name)

            rows.append(
                {
                    "vendor": self.vendor,
                    "gpu_index": gpu_index,
                    "gpu_name": name,
                    "power_w": power_w,
                    "memory_used_mib": memory_used_mib,
                    "utilization_pct": utilization_pct,
                }
            )
        return rows


class IntelSampler:
    vendor = "intel"
    POWER_SIGNAL = "GPU_POWER"
    ENERGY_SIGNAL = "GPU_ENERGY"

    def __init__(self, devices: list[int] | None):
        if shutil.which("geopmread") is None:
            raise RuntimeError("geopmread was not found")
        self.devices = devices or self._discover_devices()
        if not self.devices:
            self.devices = [0]
        self._last_energy: dict[int, tuple[float, float]] = {}

    def _discover_devices(self) -> list[int]:
        try:
            text = run_text(geopmread_cmd("-d"), env=geopmread_env())
        except Exception:
            return [0]
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "gpu":
                try:
                    return list(range(int(parts[1])))
                except ValueError:
                    return [0]
        return [0]

    def _read_signal(self, signals: tuple[str, ...], domain: str, domain_index: int) -> float:
        for signal in signals:
            try:
                return first_number(run_text(geopmread_cmd(signal, domain, domain_index), env=geopmread_env()))
            except (subprocess.SubprocessError, ValueError):
                continue
        return math.nan

    def _read_cached_signal(
        self,
        cached_signal: str | None,
        signals: tuple[str, ...],
        domain: str,
        domain_index: int,
    ) -> tuple[float, str | None]:
        if cached_signal is not None:
            try:
                return first_number(run_text(geopmread_cmd(cached_signal, domain, domain_index), env=geopmread_env())), cached_signal
            except (subprocess.SubprocessError, ValueError):
                cached_signal = None
        for signal in signals:
            try:
                return first_number(run_text(geopmread_cmd(signal, domain, domain_index), env=geopmread_env())), signal
            except (subprocess.SubprocessError, ValueError):
                continue
        return math.nan, None

    def _read_power(self, gpu_index: int) -> float:
        cmd = geopmread_cmd(self.POWER_SIGNAL, "gpu", gpu_index)
        try:
            power = power_to_watts(first_number(run_text(cmd, env=geopmread_env())))
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            print(f"Warning: {' '.join(cmd)} failed: {stderr or exc}", file=sys.stderr, flush=True)
            return self._read_power_from_energy(gpu_index)
        except subprocess.SubprocessError as exc:
            print(f"Warning: {' '.join(cmd)} failed: {exc}", file=sys.stderr, flush=True)
            return self._read_power_from_energy(gpu_index)
        if math.isnan(power):
            print(f"Warning: {' '.join(cmd)} returned nan", file=sys.stderr, flush=True)
            return self._read_power_from_energy(gpu_index)
        return power

    def _read_power_from_energy(self, gpu_index: int) -> float:
        cmd = geopmread_cmd(self.ENERGY_SIGNAL, "gpu", gpu_index)
        try:
            energy_j = first_number(run_text(cmd, env=geopmread_env()))
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            print(f"Warning: {' '.join(cmd)} failed: {stderr or exc}", file=sys.stderr, flush=True)
            return math.nan
        except subprocess.SubprocessError as exc:
            print(f"Warning: {' '.join(cmd)} failed: {exc}", file=sys.stderr, flush=True)
            return math.nan

        now = time.time()
        last = self._last_energy.get(gpu_index)
        self._last_energy[gpu_index] = (energy_j, now)
        if last is None:
            return math.nan

        last_energy_j, last_time = last
        elapsed = now - last_time
        if elapsed <= 0:
            return math.nan
        return max((energy_j - last_energy_j) / elapsed, 0.0)

    def sample(self) -> list[dict[str, object]]:
        rows = []
        for gpu_index in self.devices:
            rows.append(
                {
                    "vendor": self.vendor,
                    "gpu_index": gpu_index,
                    "gpu_name": f"geopm-gpu:{gpu_index}",
                    "power_w": self._read_power(gpu_index),
                    "memory_used_mib": math.nan,
                    "utilization_pct": math.nan,
                }
            )
        return rows


class UnavailableSampler:
    def __init__(self, vendor: str, devices: list[int] | None, reason: str):
        self.vendor = vendor
        self.devices = devices or [0]
        self.reason = reason
        print(f"Warning: {vendor} power telemetry unavailable: {reason}", file=sys.stderr, flush=True)

    def sample(self) -> list[dict[str, object]]:
        return [
            {
                "vendor": self.vendor,
                "gpu_index": gpu_index,
                "gpu_name": f"{self.vendor}:unavailable",
                "power_w": math.nan,
                "memory_used_mib": math.nan,
                "utilization_pct": math.nan,
            }
            for gpu_index in self.devices
        ]


def make_sampler(vendor: str, devices: list[int] | None):
    if vendor == "nvidia":
        return NvidiaSampler(devices)
    if vendor == "amd":
        return AmdSampler(devices)
    if vendor == "intel":
        try:
            return IntelSampler(devices)
        except RuntimeError as exc:
            return UnavailableSampler(vendor, devices, str(exc))

    errors = []
    for sampler_cls in (NvidiaSampler, AmdSampler, IntelSampler):
        try:
            return sampler_cls(devices)
        except Exception as exc:
            errors.append(f"{sampler_cls.vendor}: {exc}")
    raise RuntimeError("Could not initialize a GPU sampler: " + "; ".join(errors))


def print_detected_gpus(vendor: str, devices: list[int] | None) -> int:
    sampler = make_sampler(vendor, devices)
    try:
        for row in sampler.sample():
            print(
                ",".join(
                    [
                        str(row["vendor"]),
                        str(row["gpu_index"]),
                        str(row["gpu_name"]),
                    ]
                )
            )
    finally:
        if hasattr(sampler, "close"):
            sampler.close()
    return 0


def power_field(gpu_index: object) -> str:
    return f"GPU{gpu_index}_Power(W)"


def format_metric(value: object, digits: int = 2) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def rows_to_power_record(elapsed_s: float, rows: list[dict[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {"Time(S)": f"{elapsed_s:.2f}"}
    for row in rows:
        record[power_field(row["gpu_index"])] = format_metric(row["power_w"], 2)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", choices=("auto", "nvidia", "amd", "intel"), default="auto")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--devices", default=None, help="Comma-separated GPU indices to monitor.")
    parser.add_argument("--label", default="")
    parser.add_argument("--list-gpus", action="store_true")
    parser.add_argument("--once", action="store_true", help="Write one sample and exit.")
    args = parser.parse_args()

    devices = parse_devices(args.devices)
    if args.list_gpus:
        return print_detected_gpus(args.vendor, devices)

    if args.output is None:
        parser.error("--output is required unless --list-gpus is used")

    sampler = make_sampler(args.vendor, devices)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            first_rows = sampler.sample()
            gpu_indices = [row["gpu_index"] for row in first_rows]
            fieldnames = ["Time(S)"] + [power_field(index) for index in gpu_indices]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(rows_to_power_record(time.time() - start, first_rows))
            handle.flush()
            if args.once:
                return 0

            while True:
                now = time.time()
                writer.writerow(rows_to_power_record(now - start, sampler.sample()))
                handle.flush()
                sleep_for = args.interval - (time.time() - now)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(sampler, "close"):
            sampler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
