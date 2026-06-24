#!/usr/bin/env python
"""Create lightweight synthetic PtychoPINN inference datasets.

The generated NPZ files match the keys expected by ``PtychoDataset``. They are
intended for system-efficiency experiments, not reconstruction quality.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def make_grid_coords(count: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    side = int(np.ceil(np.sqrt(count)))
    xs = np.arange(side, dtype=np.float64) * spacing
    ys = np.arange(side, dtype=np.float64) * spacing
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return xx.reshape(-1)[:count], yy.reshape(-1)[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-images", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--object-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remake", action="store_true")
    args = parser.parse_args()

    if args.raw_images <= 0:
        raise ValueError("--raw-images must be positive")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    if args.output_dir.exists() and args.remake:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n = args.raw_images
    r = args.resolution

    # Positive diffraction intensities. Keep values modest so rounding in the
    # dataloader is stable and cheap.
    diff3d = rng.poisson(lam=20.0, size=(n, r, r)).astype(np.float32)
    label = (
        rng.random((n, r, r), dtype=np.float32)
        * np.exp(1j * rng.uniform(-np.pi, np.pi, size=(n, r, r)).astype(np.float32))
    ).astype(np.complex64)

    object_size = max(args.object_size, r * 3)
    object_guess = np.ones((object_size, object_size), dtype=np.complex64)
    probe_amp = np.exp(
        -(
            (np.linspace(-1.0, 1.0, r, dtype=np.float32)[:, None] ** 2)
            + (np.linspace(-1.0, 1.0, r, dtype=np.float32)[None, :] ** 2)
        )
        / 0.35
    )
    probe_phase = rng.uniform(-0.05, 0.05, size=(r, r)).astype(np.float32)
    probe_guess = (probe_amp * np.exp(1j * probe_phase)).astype(np.complex64)

    xcoords, ycoords = make_grid_coords(n, args.spacing)
    # Shift away from zero to look more like the existing experimental files.
    xcoords = xcoords + 32.0
    ycoords = ycoords + 32.0

    out_path = args.output_dir / f"synthetic_{n}_N{r}.npz"
    np.savez(
        out_path,
        diff3d=diff3d,
        label=label,
        objectGuess=object_guess,
        probeGuess=probe_guess,
        xcoords=xcoords,
        ycoords=ycoords,
    )
    print(out_path)
    print(f"raw_images={n}")
    print(f"resolution={r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
