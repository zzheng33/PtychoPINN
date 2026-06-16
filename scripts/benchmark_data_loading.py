#!/usr/bin/env python3
"""Benchmark PtychoPINN dataset read and memmap-write paths.

This script is intended for comparing the data-loading portion of inference
across machines that see the same dataset path.
"""

import argparse
import platform
import shutil
import socket
import tempfile
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "TP1": "data/TP1",
    "TP2": "data/TP2",
    "IC1": "data/IC1",
    "IC2": "data/IC2",
    "NCM": "data/NCM",
    "FLY1": "data/FLY1",
    "W": "data/W",
    "LFP": "data/LFP",
    "LCLS": "data/LCLS",
}


def resolve_dataset(dataset):
    path = Path(DATASETS.get(dataset, dataset))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError("Dataset directory not found: {}".format(path))
    return path


def gibibytes(num_bytes):
    return float(num_bytes) / (1024**3)


def timed(label, fn):
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    print("{}: {:.6f} s".format(label, elapsed), flush=True)
    return value, elapsed


def array_checksum(array):
    if array.size == 0:
        return 0.0
    return float(np.asarray(array).sum(dtype=np.float64))


def read_key(npz_files, key):
    checksum = 0.0
    total_bytes = 0
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            if key not in data:
                continue
            array = data[key]
            checksum += array_checksum(array)
            total_bytes += array.nbytes
    return checksum, total_bytes


def read_metadata(npz_files):
    total_keys = 0
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            total_keys += len(data.files)
            for key in ("diff3d", "xcoords", "ycoords"):
                if key in data:
                    _ = data[key].shape
    return total_keys


def write_diff3d_memmap(npz_files, write_dir):
    write_dir.mkdir(parents=True, exist_ok=True)
    checksum = 0.0
    total_bytes = 0
    for index, npz_file in enumerate(npz_files):
        with np.load(npz_file) as data:
            array = data["diff3d"]
            out_path = write_dir / "{}_{}_diff3d.dat".format(npz_file.stem, index)
            mmap = np.memmap(out_path, dtype=array.dtype, mode="w+", shape=array.shape)
            mmap[:] = array[:]
            checksum += array_checksum(mmap)
            total_bytes += mmap.nbytes
            mmap.flush()
            del mmap
    return checksum, total_bytes


def copy_npz_files(npz_files, write_dir):
    write_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for npz_file in npz_files:
        out_path = write_dir / npz_file.name
        shutil.copy2(npz_file, out_path)
        total_bytes += out_path.stat().st_size
    return total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TP1", help="Dataset key or directory path.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["diff3d", "xcoords", "ycoords", "probeGuess", "objectGuess"],
        help="NPZ array keys to fully read and sum.",
    )
    parser.add_argument(
        "--write-dir",
        type=Path,
        default=None,
        help="Directory for memmap/copy write tests. Defaults to a temp dir under /tmp.",
    )
    parser.add_argument("--skip-memmap-write", action="store_true")
    parser.add_argument("--copy-npz", action="store_true", help="Also benchmark copying NPZ files to write-dir.")
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()

    dataset_dir = resolve_dataset(args.dataset)
    npz_files = sorted(dataset_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError("No .npz files found in {}".format(dataset_dir))

    write_dir = args.write_dir
    if write_dir is None:
        write_dir = Path(tempfile.mkdtemp(prefix="ptycho_io_bench_", dir="/tmp"))
        cleanup_write_dir = not args.keep_output
    else:
        write_dir = write_dir.resolve()
        cleanup_write_dir = not args.keep_output

    file_bytes = sum(path.stat().st_size for path in npz_files)

    print("host: {}".format(socket.gethostname()))
    print("platform: {}".format(platform.platform()))
    print("python: {}".format(platform.python_version()))
    print("dataset: {}".format(dataset_dir))
    print("npz_files: {}".format(len(npz_files)))
    print("npz_file_bytes: {} ({:.3f} GiB)".format(file_bytes, gibibytes(file_bytes)))
    print("write_dir: {}".format(write_dir))
    print("repeats: {}".format(args.repeats))
    print("")

    try:
        for repeat in range(args.repeats):
            print("repeat: {}/{}".format(repeat + 1, args.repeats))

            total_keys, metadata_time = timed("metadata_time", lambda: read_metadata(npz_files))
            print("metadata_keys_seen: {}".format(total_keys))

            for key in args.keys:
                (checksum, key_bytes), elapsed = timed("read_{}_time".format(key), lambda key=key: read_key(npz_files, key))
                bandwidth = gibibytes(key_bytes) / elapsed if elapsed > 0 else float("inf")
                print("read_{}_bytes: {} ({:.3f} GiB)".format(key, key_bytes, gibibytes(key_bytes)))
                print("read_{}_bandwidth_gib_s: {:.3f}".format(key, bandwidth))
                print("read_{}_checksum: {:.6e}".format(key, checksum))

            if not args.skip_memmap_write:
                (checksum, write_bytes), elapsed = timed(
                    "memmap_write_diff3d_time",
                    lambda: write_diff3d_memmap(npz_files, write_dir / "repeat_{}_memmap".format(repeat)),
                )
                bandwidth = gibibytes(write_bytes) / elapsed if elapsed > 0 else float("inf")
                print("memmap_write_diff3d_bytes: {} ({:.3f} GiB)".format(write_bytes, gibibytes(write_bytes)))
                print("memmap_write_diff3d_bandwidth_gib_s: {:.3f}".format(bandwidth))
                print("memmap_write_diff3d_checksum: {:.6e}".format(checksum))

            if args.copy_npz:
                copied_bytes, elapsed = timed(
                    "copy_npz_time",
                    lambda: copy_npz_files(npz_files, write_dir / "repeat_{}_copy".format(repeat)),
                )
                bandwidth = gibibytes(copied_bytes) / elapsed if elapsed > 0 else float("inf")
                print("copy_npz_bytes: {} ({:.3f} GiB)".format(copied_bytes, gibibytes(copied_bytes)))
                print("copy_npz_bandwidth_gib_s: {:.3f}".format(bandwidth))

            print("")
    finally:
        if cleanup_write_dir and write_dir.exists():
            shutil.rmtree(write_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
